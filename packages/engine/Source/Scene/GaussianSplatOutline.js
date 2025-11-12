import defined from "../Core/defined.js";
import Color from "../Core/Color.js";
// import Cartesian2 from "../Core/Cartesian2.js";
import Texture from "../Renderer/Texture.js";
import PixelFormat from "../Core/PixelFormat.js";
import PixelDatatype from "../Renderer/PixelDatatype.js";
import Framebuffer from "../Renderer/Framebuffer.js";
import ClearCommand from "../Renderer/ClearCommand.js";
import PostProcessStage from "./PostProcessStage.js";
import Pass from "../Renderer/Pass.js";
import destroyObject from "../Core/destroyObject.js";

/**
 * 管理 Gaussian Splat 的轮廓渲染
 * 参考 PlayCanvas/supersplat 的 Outline 类实现 (supersplat-main/src/outline.ts)
 * 实现两阶段渲染：
 * 1. OUTLINE_PASS: 将选中的 splat 渲染到纹理
 * 2. 边缘检测: 对纹理进行 5x5 邻域检测并绘制轮廓
 *
 * @alias GaussianSplatOutline
 * @constructor
 * @param {Scene} scene Cesium 场景对象
 *
 * @private
 */
function GaussianSplatOutline(scene) {
  this._scene = scene;
  this._enabled = false;
  this._selectedTileset = undefined;

  // 状态纹理
  this._stateTexture = undefined;
  this._stateData = undefined;
  this._stateTextureWidth = 0;
  this._stateTextureHeight = 0;

  // 轮廓渲染目标
  this._outlineFramebuffer = undefined;
  this._outlineTexture = undefined;
  this._clearCommand = undefined;

  // 边缘检测 PostProcessStage
  this._edgeDetectionStage = undefined;

  // 轮廓颜色
  this._outlineColor = new Color(1.0, 0.5, 0.0, 1.0);
  this._alphaCutoff = 0.4;

  // 填充颜色和透明度（用于高亮模型内部）
  this._fillColor = new Color(1.0, 1.0, 0.0, 0.3); // 默认黄色，30%透明度
  this._fillEnabled = true; // 是否启用填充

  this._initialize();
}

Object.defineProperties(GaussianSplatOutline.prototype, {
  /**
   * 是否启用轮廓渲染
   * @memberof GaussianSplatOutline.prototype
   * @type {boolean}
   */
  enabled: {
    get: function () {
      return this._enabled;
    },
    set: function (value) {
      if (this._enabled === value) {
        console.warn("[enabled] 状态未改变");
        return;
      }
      this._enabled = value;

      if (defined(this._edgeDetectionStage)) {
        this._edgeDetectionStage.enabled = value;
      }
    },
  },

  /**
   * 轮廓颜色
   * @memberof GaussianSplatOutline.prototype
   * @type {Color}
   */
  outlineColor: {
    get: function () {
      return this._outlineColor;
    },
    set: function (value) {
      Color.clone(value, this._outlineColor);
    },
  },

  /**
   * Alpha 阈值，用于边缘检测
   * @memberof GaussianSplatOutline.prototype
   * @type {number}
   */
  alphaCutoff: {
    get: function () {
      return this._alphaCutoff;
    },
    set: function (value) {
      this._alphaCutoff = value;
    },
  },

  /**
   * 填充颜色（用于高亮模型内部）
   * @memberof GaussianSplatOutline.prototype
   * @type {Color}
   */
  fillColor: {
    get: function () {
      return this._fillColor;
    },
    set: function (value) {
      Color.clone(value, this._fillColor);
    },
  },

  /**
   * 是否启用填充
   * @memberof GaussianSplatOutline.prototype
   * @type {boolean}
   */
  fillEnabled: {
    get: function () {
      return this._fillEnabled;
    },
    set: function (value) {
      this._fillEnabled = value;
    },
  },
});

/**
 * 初始化轮廓系统
 * @private
 */
GaussianSplatOutline.prototype._initialize = function () {
  const context = this._scene.context;
  const width = context.drawingBufferWidth;
  const height = context.drawingBufferHeight;

  // 创建轮廓渲染纹理
  this._outlineTexture = new Texture({
    context: context,
    width: width,
    height: height,
    pixelFormat: PixelFormat.RGBA,
    pixelDatatype: PixelDatatype.UNSIGNED_BYTE,
  });

  // 创建 Framebuffer
  this._outlineFramebuffer = new Framebuffer({
    context: context,
    colorTextures: [this._outlineTexture],
    destroyAttachments: false,
  });

  // 创建清除命令
  this._clearCommand = new ClearCommand({
    color: new Color(0.0, 0.0, 0.0, 0.0),
    framebuffer: this._outlineFramebuffer,
  });

  // 创建边缘检测 stage
  this._createEdgeDetectionStage();
};

/**
 * 创建边缘检测 PostProcessStage
 * 参考 PlayCanvas outline-shader.ts (第 8-32 行)
 * 使用 5x5 邻域检测算法
 * @private
 */
GaussianSplatOutline.prototype._createEdgeDetectionStage = function () {
  const fragmentShader = `
    uniform sampler2D colorTexture;
    uniform sampler2D outlineTexture;
    uniform float alphaCutoff;
    uniform vec4 outlineColor;
    uniform vec4 fillColor;
    uniform bool fillEnabled;

    in vec2 v_textureCoordinates;

    void main(void) {
      // 使用整数像素坐标，与 PlayCanvas 实现一致
      // Reference: PlayCanvas outline-shader.ts line 177
      ivec2 texel = ivec2(gl_FragCoord.xy);
      vec4 originalColor = texture(colorTexture, v_textureCoordinates);
      float outlineAlpha = texelFetch(outlineTexture, texel, 0).a;

      // 检查是否为选中对象内部
      if (outlineAlpha > alphaCutoff) {
        // 选中对象内部：应用填充颜色（如果启用）
        if (fillEnabled) {
          // 混合原图像和填充颜色
          // 使用填充颜色的alpha作为混合因子
          out_FragColor = mix(originalColor, fillColor, fillColor.a);
        } else {
          // 未启用填充，输出原图像
          out_FragColor = originalColor;
        }
        return;
      }

      // 5x5 邻域边缘检测
      // Reference: PlayCanvas outline-shader.ts line 184-190
      // 注意：PlayCanvas 使用 (x != 0) && (y != 0)，但我们使用 (x != 0 || y != 0) 以包含所有邻居
      for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
          if ((x != 0 || y != 0)) {
            if (texelFetch(outlineTexture, texel + ivec2(x, y), 0).a > alphaCutoff) {
              // 找到边缘，输出轮廓颜色
              // 混合原图像和轮廓颜色
              out_FragColor = mix(originalColor, outlineColor, outlineColor.a);
              return;
            }
          }
        }
      }

      // 没有边缘，输出原图像
      out_FragColor = originalColor;
    }
  `;

  this._edgeDetectionStage = new PostProcessStage({
    name: "GaussianSplatOutlineEdgeDetection",
    fragmentShader: fragmentShader,
    uniforms: {
      outlineTexture: () => {
        // 确保纹理存在，否则返回黑色纹理避免错误
        const texture = defined(this._outlineTexture)
          ? this._outlineTexture
          : undefined;
        // 调试：检查 uniform 值
        if (defined(texture)) {
          console.log("[edgeDetection] outlineTexture uniform:", texture._id);
        } else {
          console.warn("[edgeDetection] outlineTexture 未定义！");
        }
        return texture;
      },
      alphaCutoff: () => {
        const cutoff = this._alphaCutoff;
        console.log("[edgeDetection] alphaCutoff uniform:", cutoff);
        return cutoff;
      },
      outlineColor: () => {
        const color = this._outlineColor;
        console.log("[edgeDetection] outlineColor uniform:", color);
        return color;
      },
      fillColor: () => {
        const color = this._fillColor;
        console.log("[edgeDetection] fillColor uniform:", color);
        return color;
      },
      fillEnabled: () => {
        const enabled = this._fillEnabled;
        console.log("[edgeDetection] fillEnabled uniform:", enabled);
        return enabled;
      },
    },
  });

  this._scene.postProcessStages.add(this._edgeDetectionStage);
  // 初始状态：禁用
  this._edgeDetectionStage.enabled = false;
};

/**
 * 选中 Tileset
 * 参考 PlayCanvas outline.ts 第 40-47 行 selection.changed 事件
 * @param {Cesium3DTileset} tileset 要选中的 Tileset
 */
GaussianSplatOutline.prototype.selectTileset = function (tileset) {
  if (!defined(tileset) || !defined(tileset.gaussianSplatPrimitive)) {
    return;
  }

  this._selectedTileset = tileset;
  const primitive = tileset.gaussianSplatPrimitive;

  // 初始化状态纹理
  this._initializeStateTexture(primitive);

  // 设置状态纹理引用（但不启用 outlineMode）
  // outlineMode 只在 renderOutlinePass 中临时启用
  primitive._stateTexture = this._stateTexture;
  primitive._stateTextureWidth = this._stateTextureWidth;
  primitive._stateTextureHeight = this._stateTextureHeight;

  // 启用轮廓渲染
  this.enabled = true;

  console.log("[GaussianSplatOutline] Tileset 已选中");
  console.log("- Splat 数量:", primitive._numSplats);
  console.log(
    "- 状态纹理大小:",
    this._stateTextureWidth,
    "x",
    this._stateTextureHeight,
  );
  console.log("- outlineTexture:", this._outlineTexture);
  console.log("- outlineFramebuffer:", this._outlineFramebuffer);
  console.log("- edgeDetectionStage:", this._edgeDetectionStage);
  console.log("- edgeDetectionStage.enabled 即将设置为:", true);
};

/**
 * 取消选择
 * 参考 PlayCanvas outline.ts 第 115 行 entity.enabled 控制
 */
GaussianSplatOutline.prototype.deselectTileset = function () {
  if (defined(this._selectedTileset)) {
    const primitive = this._selectedTileset.gaussianSplatPrimitive;
    if (defined(primitive)) {
      // 清除状态纹理引用
      primitive._stateTexture = undefined;

      // 确保 outlineMode 被禁用
      primitive._outlineMode = false;

      // 强制重建 draw command 以确保移除 OUTLINE_PASS
      primitive._drawCommand = undefined;
    }
  }

  this._selectedTileset = undefined;
  this.enabled = false;

  console.log("[GaussianSplatOutline] 已取消选择");
};

/**
 * 初始化状态纹理
 * 参考 PlayCanvas splat.ts 的 stateTexture 创建
 * 纹理格式：R8，每个 splat 一个字节
 * 状态位：bit 0=selected, bit 1=locked, bit 2=deleted
 * @param {GaussianSplatPrimitive} primitive
 * @private
 */
GaussianSplatOutline.prototype._initializeStateTexture = function (primitive) {
  const numSplats = primitive._numSplats;

  // 计算纹理大小（正方形布局）
  this._stateTextureWidth = Math.ceil(Math.sqrt(numSplats));
  this._stateTextureHeight = Math.ceil(numSplats / this._stateTextureWidth);

  // 创建状态数据 (所有 splat 默认为选中状态)
  const size = this._stateTextureWidth * this._stateTextureHeight;
  this._stateData = new Uint8Array(size);

  // 标记所有 splat 为选中（bit 0 = 1）
  // Reference: PlayCanvas splat.ts line 202-208
  for (let i = 0; i < numSplats; i++) {
    this._stateData[i] = 1; // STATE_SELECTED
  }

  // 调试：验证状态数据
  console.log("[状态纹理初始化] 验证状态数据:");
  console.log(
    "- 前 10 个 splat 状态:",
    Array.from(this._stateData.slice(0, 10)),
  );
  console.log(
    "- 状态数据总和:",
    this._stateData.reduce((a, b) => a + b, 0),
  );
  console.log("- 应该有", numSplats, "个 splat 被标记为选中");

  // 创建纹理
  const context = this._scene.context;

  // 如果纹理已存在且大小匹配，更新数据
  if (
    defined(this._stateTexture) &&
    this._stateTexture.width === this._stateTextureWidth &&
    this._stateTexture.height === this._stateTextureHeight
  ) {
    this._stateTexture.copyFrom({
      source: {
        arrayBufferView: this._stateData,
      },
    });
  } else {
    // 销毁旧纹理
    if (defined(this._stateTexture)) {
      this._stateTexture.destroy();
    }

    // 创建新纹理
    this._stateTexture = new Texture({
      context: context,
      width: this._stateTextureWidth,
      height: this._stateTextureHeight,
      pixelFormat: PixelFormat.RED,
      pixelDatatype: PixelDatatype.UNSIGNED_BYTE,
      source: {
        arrayBufferView: this._stateData,
      },
    });
  }
};

/**
 * 更新状态纹理
 * 参考 PlayCanvas splat.ts 第 202-208 行 updateState 方法
 * @param {Number} splatIndex Splat 索引
 * @param {Boolean} selected 是否选中
 */
GaussianSplatOutline.prototype.updateSplatState = function (
  splatIndex,
  selected,
) {
  if (!defined(this._stateData)) {
    console.warn("[updateSplatState] stateData 未定义");
    return;
  }

  const STATE_SELECTED = 1; // bit 0

  if (selected) {
    this._stateData[splatIndex] |= STATE_SELECTED;
  } else {
    this._stateData[splatIndex] &= ~STATE_SELECTED;
  }

  // 更新 GPU 纹理
  if (defined(this._stateTexture)) {
    this._stateTexture.copyFrom({
      source: {
        arrayBufferView: this._stateData,
      },
    });
  }
};

/**
 * 渲染轮廓通道
 * 参考 PlayCanvas outline.ts 第 135-160 行 postRenderLayer 事件
 * @param {FrameState} frameState 帧状态
 */
GaussianSplatOutline.prototype.renderOutlinePass = function (frameState) {
  // 只在启用且有选中对象时渲染
  if (!this._enabled || !defined(this._selectedTileset)) {
    // 清空轮廓纹理，避免显示旧内容
    if (defined(this._clearCommand)) {
      this._clearCommand.execute(frameState.context, frameState.passState);
    }
    return;
  }

  const context = frameState.context;
  const primitive = this._selectedTileset.gaussianSplatPrimitive;

  if (!defined(primitive)) {
    return;
  }

  // 清除轮廓 framebuffer
  this._clearCommand.execute(context, frameState.passState);

  // 保存原始状态
  const originalFramebuffer = context._currentFramebuffer;
  // const originalOutlineMode = primitive._outlineMode;
  const originalDrawCommand = primitive._drawCommand;

  try {
    if (!defined(originalDrawCommand)) {
      console.warn("[renderOutlinePass] 原始 drawCommand 不存在，跳过");
      return;
    }

    console.log("[renderOutlinePass] 开始渲染轮廓通道");

    // 临时启用 OUTLINE_PASS（在打印之前）
    primitive._outlineMode = true;

    console.log("- outlineMode 已设置为:", primitive._outlineMode);
    console.log("- stateTexture:", primitive._stateTexture);
    console.log("- stateTexture defined:", defined(primitive._stateTexture));
    console.log(
      "- 检查条件:",
      primitive._outlineMode && defined(primitive._stateTexture),
    );

    // 关键：强制重建 draw command 以应用 OUTLINE_PASS define
    // 必须清空并调用 buildGSplatDrawCommand
    primitive._drawCommand = undefined;

    console.log("准备调用 buildGSplatDrawCommand:");
    console.log("- _outlineMode:", primitive._outlineMode);
    console.log("- _stateTexture:", primitive._stateTexture);
    console.log("- _stateTextureWidth:", primitive._stateTextureWidth);
    console.log("- _stateTextureHeight:", primitive._stateTextureHeight);

    // 直接调用 buildGSplatDrawCommand 重建
    const GaussianSplatPrimitive = primitive.constructor;
    GaussianSplatPrimitive.buildGSplatDrawCommand(primitive, frameState);

    console.log("- drawCommand 重建后:", primitive._drawCommand);

    // 检查重建后的 shader
    if (
      defined(primitive._drawCommand) &&
      defined(primitive._drawCommand.shaderProgram)
    ) {
      const sp = primitive._drawCommand.shaderProgram;
      console.log("- shader program:", sp);
      console.log(
        "- uniformMap 包含 u_splatStateTexture:",
        "u_splatStateTexture" in primitive._drawCommand.uniformMap,
      );
    }

    if (!defined(primitive._drawCommand)) {
      console.error("- drawCommand 重建失败！");
      return;
    }

    // 切换到轮廓 framebuffer
    context._currentFramebuffer = this._outlineFramebuffer;

    // 清除 framebuffer（重要！）
    this._clearCommand.execute(context, frameState.passState);

    // 执行渲染命令（应该使用 OUTLINE_PASS shader）
    console.log("- 执行渲染命令到 outlineTexture");
    console.log("- 当前 framebuffer:", context._currentFramebuffer);
    console.log("- outlineFramebuffer:", this._outlineFramebuffer);
    console.log(
      "- outlineTexture 尺寸:",
      this._outlineTexture.width,
      "x",
      this._outlineTexture.height,
    );
    console.log(
      "- drawCommand renderState:",
      primitive._drawCommand.renderState,
    );
    console.log(
      "- drawCommand instanceCount:",
      primitive._drawCommand.instanceCount,
    );
    console.log("- drawCommand count:", primitive._drawCommand.count);
    console.log("- drawCommand pass:", primitive._drawCommand.pass);
    console.log("- frameState.pass:", frameState.pass);

    // 检查 drawCommand 是否有效
    if (!defined(primitive._drawCommand.vertexArray)) {
      console.error("- drawCommand.vertexArray 未定义！");
      return;
    }
    if (!defined(primitive._drawCommand.shaderProgram)) {
      console.error("- drawCommand.shaderProgram 未定义！");
      return;
    }

    // 检查视口
    console.log("- czm_viewport:", frameState.context.uniformState.viewport);
    console.log("- drawingBufferWidth:", context.drawingBufferWidth);
    console.log("- drawingBufferHeight:", context.drawingBufferHeight);

    // 临时：强制设置 pass 为 GAUSSIAN_SPLATS
    const originalPass = primitive._drawCommand.pass;
    primitive._drawCommand.pass = Pass.GAUSSIAN_SPLATS;

    // 创建自定义 passState，确保 pass 匹配
    // 需要复制 frameState.passState 的所有属性，然后覆盖必要的属性
    const customPassState = Object.assign({}, frameState.passState, {
      framebuffer: this._outlineFramebuffer,
      pass: Pass.GAUSSIAN_SPLATS,
      context: context, // 重要：需要 context 属性
    });

    // 更新 uniformState 的 pass
    const originalUniformPass = frameState.context.uniformState._pass;
    frameState.context.uniformState.updatePass(Pass.GAUSSIAN_SPLATS);

    try {
      console.log("- 使用自定义 passState，pass:", customPassState.pass);
      primitive._drawCommand.execute(context, customPassState);
      console.log("- 渲染完成");
    } catch (error) {
      console.error("- 渲染执行错误:", error);
      throw error;
    } finally {
      // 恢复原 pass
      primitive._drawCommand.pass = originalPass;
      frameState.context.uniformState.updatePass(originalUniformPass);
    }

    // 调试：检查渲染后的纹理（通过读取像素）
    // 注意：这可能会影响性能，仅用于调试
    if (context._gl && this._outlineTexture._texture) {
      const gl = context._gl;
      const tempFramebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, tempFramebuffer);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        this._outlineTexture._texture,
        0,
      );

      if (
        gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE
      ) {
        const pixels = new Uint8Array(4);
        gl.readPixels(
          Math.floor(this._outlineTexture.width / 2),
          Math.floor(this._outlineTexture.height / 2),
          1,
          1,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          pixels,
        );
        console.log("- outlineTexture 中心像素 (RGBA):", pixels);
        console.log("- outlineTexture 中心像素 alpha:", pixels[3] / 255.0);
      }

      gl.deleteFramebuffer(tempFramebuffer);
      gl.bindFramebuffer(
        gl.FRAMEBUFFER,
        context._currentFramebuffer
          ? context._currentFramebuffer._framebuffer
          : null,
      );
    }
  } catch (error) {
    console.error("[renderOutlinePass] 渲染错误:", error);
  } finally {
    // 恢复原 framebuffer
    context._currentFramebuffer = originalFramebuffer;

    // 关键：禁用 outlineMode 并恢复原 draw command
    primitive._outlineMode = false;
    primitive._drawCommand = originalDrawCommand;

    console.log("[renderOutlinePass] 轮廓通道渲染完成");
    console.log("- outlineMode 已恢复为:", primitive._outlineMode);
    console.log(
      "- edgeDetectionStage.enabled:",
      this._edgeDetectionStage.enabled,
    );
  }
};

/**
 * 更新轮廓系统（在每帧渲染时调用）
 * @param {FrameState} frameState 帧状态
 */
GaussianSplatOutline.prototype.update = function (frameState) {
  if (!this._enabled) {
    return;
  }

  // 检查窗口大小变化，需要重建纹理
  const context = frameState.context;
  const width = context.drawingBufferWidth;
  const height = context.drawingBufferHeight;

  if (
    this._outlineTexture.width !== width ||
    this._outlineTexture.height !== height
  ) {
    // 重建纹理
    this._outlineTexture.destroy();
    this._outlineFramebuffer.destroy();

    this._outlineTexture = new Texture({
      context: context,
      width: width,
      height: height,
      pixelFormat: PixelFormat.RGBA,
      pixelDatatype: PixelDatatype.UNSIGNED_BYTE,
    });

    this._outlineFramebuffer = new Framebuffer({
      context: context,
      colorTextures: [this._outlineTexture],
      destroyAttachments: false,
    });

    this._clearCommand.framebuffer = this._outlineFramebuffer;
  }
};

/**
 * 判断是否已销毁
 * @returns {boolean}
 */
GaussianSplatOutline.prototype.isDestroyed = function () {
  return false;
};

/**
 * 销毁资源
 * @returns {*}
 */
GaussianSplatOutline.prototype.destroy = function () {
  if (defined(this._stateTexture)) {
    this._stateTexture.destroy();
    this._stateTexture = undefined;
  }
  if (defined(this._outlineTexture)) {
    this._outlineTexture.destroy();
    this._outlineTexture = undefined;
  }
  if (defined(this._outlineFramebuffer)) {
    this._outlineFramebuffer.destroy();
    this._outlineFramebuffer = undefined;
  }
  if (defined(this._edgeDetectionStage)) {
    this._scene.postProcessStages.remove(this._edgeDetectionStage);
    this._edgeDetectionStage = undefined;
  }

  this._stateData = undefined;

  return destroyObject(this);
};

export default GaussianSplatOutline;
