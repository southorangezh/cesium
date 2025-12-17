import Cartesian2 from "../Core/Cartesian2.js";
import Frozen from "../Core/Frozen.js";
import Matrix4 from "../Core/Matrix4.js";
import ModelUtility from "./Model/ModelUtility.js";
import GaussianSplatSorter from "./GaussianSplatSorter.js";
import GaussianSplatTextureGenerator from "./GaussianSplatTextureGenerator.js";
import ComponentDatatype from "../Core/ComponentDatatype.js";
import PixelDatatype from "../Renderer/PixelDatatype.js";
import PixelFormat from "../Core/PixelFormat.js";
import Sampler from "../Renderer/Sampler.js";
import Texture from "../Renderer/Texture.js";
import FramebufferManager from "../Renderer/FramebufferManager.js";
import GaussianSplatRenderResources from "./GaussianSplatRenderResources.js";
import BlendingState from "./BlendingState.js";
import Pass from "../Renderer/Pass.js";
import ShaderDestination from "../Renderer/ShaderDestination.js";
import GaussianSplatVS from "../Shaders/PrimitiveGaussianSplatVS.js";
import GaussianSplatFS from "../Shaders/PrimitiveGaussianSplatFS.js";
import PrimitiveType from "../Core/PrimitiveType.js";
import DrawCommand from "../Renderer/DrawCommand.js";
import Geometry from "../Core/Geometry.js";
import GeometryAttribute from "../Core/GeometryAttribute.js";
import VertexArray from "../Renderer/VertexArray.js";
import BufferUsage from "../Renderer/BufferUsage.js";
import RenderState from "../Renderer/RenderState.js";
import ClearCommand from "../Renderer/ClearCommand.js";
import ShaderSource from "../Renderer/ShaderSource.js";
import clone from "../Core/clone.js";
import defined from "../Core/defined.js";
import VertexAttributeSemantic from "./VertexAttributeSemantic.js";
import AttributeType from "./AttributeType.js";
import ModelComponents from "./ModelComponents.js";
import Axis from "./Axis.js";
import Cartesian3 from "../Core/Cartesian3.js";
import Quaternion from "../Core/Quaternion.js";
import SplitDirection from "./SplitDirection.js";
import destroyObject from "../Core/destroyObject.js";
import ContextLimits from "../Renderer/ContextLimits.js";
import Transforms from "../Core/Transforms.js";
import Color from "../Core/Color.js";
import CesiumMath from "../Core/Math.js";
import SceneTransforms from "./SceneTransforms.js";

const scratchMatrix4A = new Matrix4();
const scratchMatrix4B = new Matrix4();
const scratchMatrix4C = new Matrix4();
const scratchMatrix4D = new Matrix4();
const scratchFilterPosition = new Cartesian3();
const scratchFilterRgba = new Uint8Array(4);
const scratchFilterColor = new Color();
const scratchPickDrawingBuffer = new Cartesian2();
const scratchPickProjected = new Cartesian2();
const scratchPickCartesian = new Cartesian3();
const scratchPickWorldPosition = new Cartesian3();
const outlineClearColor = new Color(0.0, 0.0, 0.0, 0.0);

const OutlineCompositeFS = `
uniform sampler2D u_outlineMaskTexture;
uniform sampler2D u_outlineDepthTexture;
uniform bool u_outlineHasDepth;
uniform bool u_outlineDepthTest;
uniform vec4 u_outlineColor;
uniform float u_outlineAlphaCutoff;
uniform float u_outlineMinAlphaDiff;
uniform float u_outlineWidth;
uniform vec2 u_outlineTexelStep;
uniform int u_outlineKernelRadius;
uniform bool u_outlineDebugMode;

const int MAX_KERNEL_RADIUS = 4;

void main()
{
    vec2 uv = gl_FragCoord.xy / czm_viewport.zw;
    float centerAlpha = texture(u_outlineMaskTexture, uv).a;
    bool isRingsMode = u_outlineAlphaCutoff < 0.05;
    float widthBias = clamp((u_outlineWidth - 1.5) * 0.05, -0.2, 0.3);
    float centerThreshold = isRingsMode
        ? clamp(0.25 - widthBias, 0.02, 0.6)
        : clamp(u_outlineAlphaCutoff * 0.5 - widthBias, 0.02, 0.9);
    float edgeThreshold = isRingsMode
        ? clamp(0.65 - widthBias, 0.1, 0.95)
        : clamp(u_outlineAlphaCutoff + widthBias, 0.05, 0.98);
    if (centerAlpha >= centerThreshold)
    {
        discard;
    }

    // 深度辅助：仅当 mask 深度未被其它几何遮挡时才绘制
    float sceneDepth = czm_unpackDepth(texture(czm_globeDepthTexture, uv));
    float depthFade = 1.0;
    if (u_outlineDepthTest && u_outlineHasDepth && sceneDepth > 0.0)
    {
        float maskDepth = texture(u_outlineDepthTexture, uv).r;
        if (maskDepth > 0.0 && maskDepth < 1.0)
        {
            vec4 sceneWindow = vec4(gl_FragCoord.xy, sceneDepth, 1.0);
            vec4 maskWindow = vec4(gl_FragCoord.xy, maskDepth, 1.0);
            float sceneEyeDepth = -czm_windowToEyeCoordinates(sceneWindow).z;
            float maskEyeDepth = -czm_windowToEyeCoordinates(maskWindow).z;

            const float depthSoftThreshold = 0.008;
            const float depthHardLimit = 0.03;
            float depthDelta = maskEyeDepth - sceneEyeDepth;
            if (depthDelta > depthHardLimit)
            {
                discard;
            }
            depthFade = 1.0 - clamp(max(depthDelta - depthSoftThreshold, 0.0) / (depthHardLimit - depthSoftThreshold), 0.0, 1.0);
        }
    }

    // 邻域采样，使用浮点半径进行平滑滤波
    bool isEdge = false;
    float radiusOffset = clamp((u_outlineWidth - 1.5) * 0.5, -1.0, 2.0);
    float effectiveRadius = clamp(float(u_outlineKernelRadius) + radiusOffset, 0.75, float(MAX_KERNEL_RADIUS));
    float effectiveRadiusSq = effectiveRadius * effectiveRadius;
    float maxNeighborAlpha = 0.0;
    float gradientMax = 0.0;

    for (int x = -MAX_KERNEL_RADIUS; x <= MAX_KERNEL_RADIUS; ++x)
    {
        if (abs(x) > MAX_KERNEL_RADIUS)
        {
            continue;
        }
        for (int y = -MAX_KERNEL_RADIUS; y <= MAX_KERNEL_RADIUS; ++y)
        {
            if (abs(y) > MAX_KERNEL_RADIUS)
            {
                continue;
            }
            float distanceSq = float(x * x + y * y);
            if (distanceSq > effectiveRadiusSq)
            {
                continue;
            }
            // 跳过中心像素
            if (x == 0 && y == 0)
            {
                continue;
            }

            vec2 offset = vec2(float(x), float(y)) * u_outlineTexelStep;
            float neighborAlpha = texture(u_outlineMaskTexture, uv + offset).a;

            maxNeighborAlpha = max(maxNeighborAlpha, neighborAlpha);
            gradientMax = max(gradientMax, abs(neighborAlpha - centerAlpha));
        }
        if (isEdge)
        {
            break;
        }
    }

    if (isRingsMode)
    {
        isEdge = (maxNeighborAlpha >= edgeThreshold) && (centerAlpha <= centerThreshold);
    }
    else
    {
        float gradientThreshold = max(u_outlineMinAlphaDiff, 0.02);
        isEdge = (maxNeighborAlpha >= edgeThreshold &&
                  centerAlpha <= centerThreshold &&
                  gradientMax >= gradientThreshold);
    }

    if (!isEdge)
    {
        discard;
    }

    // 调试模式：可视化边缘检测结果
    if (u_outlineDebugMode)
    {
        // 使用颜色编码调试信息
        // R: 中心 alpha 值
        // G: 检测到的最大邻域 alpha 值
        // B: alpha 差值
        float alphaDiff = maxNeighborAlpha - centerAlpha;
        out_FragColor = vec4(centerAlpha, maxNeighborAlpha, alphaDiff, depthFade);
    }
    else
    {
        // 正常模式：根据深度淡出输出轮廓颜色
        out_FragColor = vec4(u_outlineColor.rgb, u_outlineColor.a * depthFade);
    }
}
`;

const GaussianSplatSortingState = {
  IDLE: 0,
  WAITING: 1,
  SORTING: 2,
  SORTED: 3,
  ERROR: 4,
};

function createSphericalHarmonicsTexture(context, shData) {
  const texture = new Texture({
    context: context,
    source: {
      width: shData.width,
      height: shData.height,
      arrayBufferView: shData.data,
    },
    preMultiplyAlpha: false,
    skipColorSpaceConversion: true,
    pixelFormat: PixelFormat.RG_INTEGER,
    pixelDatatype: PixelDatatype.UNSIGNED_INT,
    flipY: false,
    sampler: Sampler.NEAREST,
  });

  return texture;
}

function createGaussianSplatTexture(context, splatTextureData) {
  return new Texture({
    context: context,
    source: {
      width: splatTextureData.width,
      height: splatTextureData.height,
      arrayBufferView: splatTextureData.data,
    },
    preMultiplyAlpha: false,
    skipColorSpaceConversion: true,
    pixelFormat: PixelFormat.RGBA_INTEGER,
    pixelDatatype: PixelDatatype.UNSIGNED_INT,
    flipY: false,
    sampler: Sampler.NEAREST,
  });
}

function createSplatStateTexture(context, stateData, width, height) {
  // Use non-integer format for single-channel byte data
  // WebGL2: RED, WebGL1: LUMINANCE
  const pixelFormat = context.webgl2 ? PixelFormat.RED : PixelFormat.LUMINANCE;

  return new Texture({
    context: context,
    source: {
      width: width,
      height: height,
      arrayBufferView: stateData,
    },
    preMultiplyAlpha: false,
    skipColorSpaceConversion: true,
    pixelFormat: pixelFormat,
    pixelDatatype: PixelDatatype.UNSIGNED_BYTE,
    flipY: false,
    sampler: Sampler.NEAREST,
  });
}

function clampByte(value) {
  return Math.round(CesiumMath.clamp(value, 0.0, 255.0));
}

function assignArrayColor(target, source) {
  const length = Math.min(source.length, 4);
  for (let i = 0; i < length; i++) {
    const component = source[i];
    target[i] =
      component <= 1.0 && component >= 0.0
        ? clampByte(component * 255.0)
        : clampByte(component);
  }
}

function updateOutlineFramebuffer(primitive, frameState) {
  if (!defined(primitive._outlineFramebuffer)) {
    primitive._outlineFramebuffer = new FramebufferManager({
      depth: true,
      supportsDepthTexture: true,
    });
  }

  const context = frameState.context;
  const width = context.drawingBufferWidth;
  const height = context.drawingBufferHeight;

  primitive._outlineFramebuffer.update(
    context,
    width,
    height,
    undefined,
    undefined,
    PixelFormat.RGBA,
  );

  primitive._outlineMaskTexture =
    primitive._outlineFramebuffer.getColorTexture(0);
  primitive._outlineDepthTexture =
    primitive._outlineFramebuffer.getDepthTexture();

  if (!defined(primitive._outlineClearCommand)) {
    primitive._outlineClearCommand = new ClearCommand({
      color: outlineClearColor,
      depth: 1.0,
      renderState: RenderState.fromCache(),
      pass: Pass.OPAQUE,
      owner: primitive,
    });
  }

  primitive._outlineClearCommand.framebuffer =
    primitive._outlineFramebuffer.framebuffer;
}

function destroyOutlineResources(primitive) {
  if (defined(primitive._outlineFramebuffer)) {
    primitive._outlineFramebuffer.destroy();
    primitive._outlineFramebuffer = undefined;
  }
  primitive._outlineMaskTexture = undefined;
  primitive._outlineDepthTexture = undefined;
  primitive._outlineMaskCommand = undefined;
  primitive._outlineCompositeCommand = undefined;
  primitive._outlineClearCommand = undefined;
  if (defined(primitive._boundaryVertexArray)) {
    primitive._boundaryVertexArray.destroy();
    primitive._boundaryVertexArray = undefined;
  }
  primitive._boundaryIndices = undefined;
  primitive._boundaryInstanceCount = 0;
  primitive._outlineRingCommand = undefined;
  primitive._outlineBoundaryStats = undefined;
}

function createOutlineMaskUniformMap(baseUniformMap) {
  const uniformMap = {};
  for (const key in baseUniformMap) {
    if (
      Object.prototype.hasOwnProperty.call(baseUniformMap, key) &&
      key !== "u_outlineMaskPass"
    ) {
      uniformMap[key] = baseUniformMap[key];
    }
  }
  uniformMap.u_outlineMaskPass = function () {
    return true;
  };
  return uniformMap;
}

function createOutlineRingUniformMap(baseUniformMap) {
  const uniformMap = {};
  for (const key in baseUniformMap) {
    if (!Object.prototype.hasOwnProperty.call(baseUniformMap, key)) {
      continue;
    }
    if (key === "u_outlineRingPass") {
      uniformMap[key] = function () {
        return true;
      };
    } else if (key === "u_outlineMaskPass") {
      uniformMap[key] = function () {
        return false;
      };
    } else {
      uniformMap[key] = baseUniformMap[key];
    }
  }
  if (!defined(uniformMap.u_outlineRingPass)) {
    uniformMap.u_outlineRingPass = function () {
      return true;
    };
  }
  return uniformMap;
}

GaussianSplatPrimitive.prototype._computeBoundaryIndices = function () {
  const neighborRange = CesiumMath.clamp(
    Math.round(this._outlineBoundaryNeighborRange),
    1,
    4,
  );
  const coverageThreshold = CesiumMath.clamp(
    this._outlineBoundaryMinCoverage,
    0.0,
    1.0,
  );

  if (this._selectedSplatIndices.size === 0) {
    this._outlineBoundaryStats = {
      selectedCount: 0,
      boundaryCount: 0,
      cellSize: 0,
      cellMultiplier: this._outlineBoundaryCellMultiplier,
      minCellSize: this._outlineBoundaryMinCellSize,
      gridCellCount: 0,
      neighborRange: neighborRange,
      coverageThreshold: coverageThreshold,
      fallback: false,
    };
    return new Uint32Array(0);
  }

  if (!defined(this._positions)) {
    const typed = Uint32Array.from(this._selectedSplatIndices);
    this._outlineBoundaryStats = {
      selectedCount: typed.length,
      boundaryCount: typed.length,
      cellSize: 0,
      cellMultiplier: this._outlineBoundaryCellMultiplier,
      minCellSize: this._outlineBoundaryMinCellSize,
      gridCellCount: 0,
      neighborRange: neighborRange,
      coverageThreshold: coverageThreshold,
      fallback: true,
    };
    return typed;
  }

  const selectedArray = Array.from(this._selectedSplatIndices);
  const positions = this._positions;
  const scales = this._scales;

  let averageScale = 0.0;
  let scaleSamples = 0;
  if (defined(scales)) {
    for (let i = 0; i < selectedArray.length; i++) {
      const index = selectedArray[i];
      const sx = scales[index * 3];
      const sy = scales[index * 3 + 1];
      const maxScale = Math.max(Math.abs(sx) || 0.0, Math.abs(sy) || 0.0);
      if (maxScale > 0.0 && isFinite(maxScale)) {
        averageScale += maxScale;
        scaleSamples++;
      }
    }
  }
  averageScale =
    scaleSamples > 0
      ? averageScale / scaleSamples
      : Math.max(this._outlineWidth, 1.0);

  const cellMultiplier = Math.max(this._outlineBoundaryCellMultiplier, 0.1);
  const minCellSize = Math.max(this._outlineBoundaryMinCellSize, 0.01);
  const cellSize = Math.max(averageScale * cellMultiplier, minCellSize);
  const grid = new Map();

  for (let i = 0; i < selectedArray.length; i++) {
    const index = selectedArray[i];
    const px = positions[index * 3];
    const py = positions[index * 3 + 1];
    const cellX = Math.floor(px / cellSize);
    const cellY = Math.floor(py / cellSize);
    const key = `${cellX},${cellY}`;

    let cell = grid.get(key);
    if (!cell) {
      cell = {
        indices: [],
        x: cellX,
        y: cellY,
      };
      grid.set(key, cell);
    }
    cell.indices.push(index);
  }

  const boundaryList = [];
  const totalNeighborSlots =
    (neighborRange * 2 + 1) * (neighborRange * 2 + 1) - 1;
  grid.forEach((cell) => {
    let neighborHits = 0;
    for (let dx = -neighborRange; dx <= neighborRange; dx++) {
      for (let dy = -neighborRange; dy <= neighborRange; dy++) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const neighborKey = `${cell.x + dx},${cell.y + dy}`;
        if (grid.has(neighborKey)) {
          neighborHits++;
        }
      }
    }
    const coverage =
      totalNeighborSlots > 0 ? neighborHits / totalNeighborSlots : 0.0;
    if (coverage < coverageThreshold) {
      for (let i = 0; i < cell.indices.length; i++) {
        boundaryList.push(cell.indices[i]);
      }
    }
  });

  const usedFallback = boundaryList.length === 0;
  const boundaryArray = usedFallback ? selectedArray : boundaryList;
  const typedArray = Uint32Array.from(boundaryArray);
  this._outlineBoundaryStats = {
    selectedCount: selectedArray.length,
    boundaryCount: typedArray.length,
    cellSize: cellSize,
    cellMultiplier: cellMultiplier,
    minCellSize: minCellSize,
    gridCellCount: grid.size,
    neighborRange: neighborRange,
    coverageThreshold: coverageThreshold,
    fallback: usedFallback,
  };

  return typedArray;
};

GaussianSplatPrimitive.prototype._ensureBoundaryResources = function (
  frameState,
) {
  if (!this._boundaryDirty && defined(this._boundaryVertexArray)) {
    return;
  }

  this._boundaryDirty = false;
  const indices = this._computeBoundaryIndices();
  this._boundaryIndices = indices;
  this._boundaryInstanceCount = indices.length;

  if (this._boundaryInstanceCount === 0) {
    if (defined(this._boundaryVertexArray)) {
      this._boundaryVertexArray.destroy();
      this._boundaryVertexArray = undefined;
    }
    return;
  }

  const geometry = new Geometry({
    attributes: {
      screenQuadPosition: new GeometryAttribute({
        componentDatatype: ComponentDatatype.FLOAT,
        componentsPerAttribute: 2,
        values: [-1, -1, 1, -1, 1, 1, -1, 1],
        name: "_SCREEN_QUAD_POS",
        variableName: "screenQuadPosition",
      }),
      splatIndex: {
        componentDatatype: ComponentDatatype.UNSIGNED_INT,
        componentsPerAttribute: 1,
        values: indices,
        variableName: "splatIndex",
        instanceDivisor: 1,
      },
    },
    primitiveType: PrimitiveType.TRIANGLE_STRIP,
  });

  if (defined(this._boundaryVertexArray)) {
    this._boundaryVertexArray.destroy();
  }

  this._boundaryVertexArray = VertexArray.fromGeometry({
    context: frameState.context,
    geometry: geometry,
    attributeLocations: {
      screenQuadPosition: 0,
      splatIndex: 2,
    },
    bufferUsage: BufferUsage.DYNAMIC_DRAW,
    interleave: false,
  });
};

function applyCustomColorFilters(
  primitive,
  colors,
  componentsPerColor,
  isRgba,
) {
  if (
    !defined(primitive._customColorFilters) ||
    primitive._customColorFilters.length === 0
  ) {
    return;
  }

  const activeFilters = primitive._customColorFilters.filter(
    (filter) =>
      defined(filter) &&
      filter.enabled !== false &&
      typeof filter.callback === "function",
  );

  if (activeFilters.length === 0) {
    return;
  }

  activeFilters.sort((a, b) => (a.priority || 0) - (b.priority || 0));

  const hasPositions = defined(primitive._positions);
  const positions = primitive._positions;
  const hasPlyIndices = defined(primitive._plyIndicesAggregate);
  const contextPosition = hasPositions ? scratchFilterPosition : undefined;

  const filterContext = {
    tileset: primitive._tileset,
    primitive: primitive,
    aggregateIndex: 0,
    plyIndex: 0,
    position: contextPosition,
    rgba: scratchFilterRgba,
    color: scratchFilterColor,
    userData: undefined,
  };

  for (let i = 0; i < primitive._numSplats; i++) {
    const colorIndex = i * componentsPerColor;
    scratchFilterRgba[0] = colors[colorIndex];
    scratchFilterRgba[1] = colors[colorIndex + 1];
    scratchFilterRgba[2] = colors[colorIndex + 2];
    scratchFilterRgba[3] = isRgba ? colors[colorIndex + 3] : 255;

    Color.fromBytes(
      scratchFilterRgba[0],
      scratchFilterRgba[1],
      scratchFilterRgba[2],
      scratchFilterRgba[3],
      scratchFilterColor,
    );

    filterContext.aggregateIndex = i;
    filterContext.plyIndex = hasPlyIndices
      ? primitive._plyIndicesAggregate[i]
      : i;

    if (hasPositions && defined(contextPosition)) {
      const positionIndex = i * 3;
      contextPosition.x = positions[positionIndex];
      contextPosition.y = positions[positionIndex + 1];
      contextPosition.z = positions[positionIndex + 2];
    }

    let modified = false;
    let overrideAlphaSet = false;
    let overrideAlphaValue = 255;

    for (let j = 0; j < activeFilters.length; j++) {
      const descriptor = activeFilters[j];
      filterContext.userData = descriptor.userData;

      let result;
      try {
        result = descriptor.callback.call(
          descriptor.thisArg,
          filterContext,
          descriptor.options,
        );
      } catch (error) {
        console.error(
          `Gaussian splat filter "${
            descriptor.name || descriptor.id || "anonymous"
          }" failed:`,
          error,
        );
        continue;
      }

      if (!defined(result) || result === false) {
        continue;
      }

      let stopProcessing = false;

      if (result.discard === true) {
        scratchFilterRgba[3] = 0;
        modified = true;
        stopProcessing = true;
      }

      if (defined(result.alpha)) {
        const alphaValue =
          result.alpha > 1.0 ? result.alpha : result.alpha * 255.0;
        overrideAlphaValue = clampByte(alphaValue);
        scratchFilterRgba[3] = overrideAlphaValue;
        overrideAlphaSet = true;
        modified = true;
      }

      if (defined(result.multiply)) {
        const factor = result.multiply;
        scratchFilterRgba[0] = clampByte(scratchFilterRgba[0] * factor);
        scratchFilterRgba[1] = clampByte(scratchFilterRgba[1] * factor);
        scratchFilterRgba[2] = clampByte(scratchFilterRgba[2] * factor);
        modified = true;
      }

      if (defined(result.rgba)) {
        assignArrayColor(scratchFilterRgba, result.rgba);
        if (overrideAlphaSet) {
          scratchFilterRgba[3] = overrideAlphaValue;
        }
        modified = true;
      }

      if (defined(result.color)) {
        // 保存原始颜色用于混合
        const originalRgba = [
          scratchFilterRgba[0],
          scratchFilterRgba[1],
          scratchFilterRgba[2],
          scratchFilterRgba[3],
        ];

        let filterColorRgba;
        if (result.color instanceof Color) {
          filterColorRgba = new Uint8Array(4);
          result.color.toBytes(filterColorRgba);
        } else if (Array.isArray(result.color)) {
          filterColorRgba = new Uint8Array(4);
          assignArrayColor(filterColorRgba, result.color);
        } else {
          filterColorRgba = scratchFilterRgba;
        }

        // 如果指定了 filterStrength，进行颜色混合（参考着色器逻辑）
        if (defined(result.filterStrength)) {
          const filterStrength = CesiumMath.clamp(
            result.filterStrength,
            0.0,
            1.0,
          );

          // 参考着色器逻辑：mix(originalColor, colorFiltered, filterStrength)
          // 增强匹配滤镜颜色的通道，减少其他通道
          const colorFiltered = [
            originalRgba[0] * (0.5 + (filterColorRgba[0] / 255.0) * 0.9),
            originalRgba[1] * (0.5 + (filterColorRgba[1] / 255.0) * 0.9),
            originalRgba[2] * (0.5 + (filterColorRgba[2] / 255.0) * 0.9),
          ];

          // 混合原始颜色和滤镜颜色
          scratchFilterRgba[0] = clampByte(
            originalRgba[0] * (1.0 - filterStrength) +
              colorFiltered[0] * filterStrength,
          );
          scratchFilterRgba[1] = clampByte(
            originalRgba[1] * (1.0 - filterStrength) +
              colorFiltered[1] * filterStrength,
          );
          scratchFilterRgba[2] = clampByte(
            originalRgba[2] * (1.0 - filterStrength) +
              colorFiltered[2] * filterStrength,
          );
        } else {
          // 没有 filterStrength，直接应用滤镜颜色
          scratchFilterRgba[0] = filterColorRgba[0];
          scratchFilterRgba[1] = filterColorRgba[1];
          scratchFilterRgba[2] = filterColorRgba[2];
        }

        if (overrideAlphaSet) {
          scratchFilterRgba[3] = overrideAlphaValue;
        } else if (
          result.color instanceof Color ||
          Array.isArray(result.color)
        ) {
          scratchFilterRgba[3] = filterColorRgba[3];
        }
        modified = true;
      }

      if (defined(result.tint)) {
        const tint = CesiumMath.clamp(result.tint, 0.0, 1.0);
        scratchFilterRgba[0] = clampByte(
          scratchFilterRgba[0] * (1.0 - tint) + 255.0 * tint,
        );
        scratchFilterRgba[1] = clampByte(
          scratchFilterRgba[1] * (1.0 - tint) + 255.0 * tint,
        );
        scratchFilterRgba[2] = clampByte(
          scratchFilterRgba[2] * (1.0 - tint) + 255.0 * tint,
        );
        modified = true;
      }

      Color.fromBytes(
        scratchFilterRgba[0],
        scratchFilterRgba[1],
        scratchFilterRgba[2],
        scratchFilterRgba[3],
        scratchFilterColor,
      );

      if (descriptor.once === true) {
        descriptor.enabled = false;
      }

      if (defined(result.stop) ? result.stop : stopProcessing) {
        break;
      }
    }

    if (modified) {
      colors[colorIndex] = scratchFilterRgba[0];
      colors[colorIndex + 1] = scratchFilterRgba[1];
      colors[colorIndex + 2] = scratchFilterRgba[2];
      if (isRgba) {
        colors[colorIndex + 3] = scratchFilterRgba[3];
      }
    }
  }
}

/** A primitive that renders Gaussian splats.
 * <p>
 * This primitive is used to render Gaussian splats in a 3D Tileset.
 * It is designed to work with the KHR_gaussian_splatting and KHR_gaussian_splatting_compression_spz_2 extensions.
 * </p>
 * @alias GaussianSplatPrimitive
 * @constructor
 * @param {object} options An object with the following properties:
 * @param {Cesium3DTileset} options.tileset The tileset that this primitive belongs to.
 * @param {boolean} [options.debugShowBoundingVolume=false] Whether to show the bounding volume of the primitive for debugging purposes.
 * @private
 */

function GaussianSplatPrimitive(options) {
  options = options ?? Frozen.EMPTY_OBJECT;

  /**
   * The positions of the Gaussian splats in the primitive.
   * @type {undefined|Float32Array}
   * @private
   */
  this._positions = undefined;
  /**
   * The rotations of the Gaussian splats in the primitive.
   * @type {undefined|Float32Array}
   * @private
   */
  this._rotations = undefined;
  /**
   * The scales of the Gaussian splats in the primitive.
   * @type {undefined|Float32Array}
   * @private
   */
  this._scales = undefined;
  /**
   * The colors of the Gaussian splats in the primitive.
   * @type {undefined|Uint8Array}
   * @private
   */
  this._colors = undefined;
  /**
   * The indexes of the Gaussian splats in the primitive.
   * Used to index into the splat attribute texture in the vertex shader.
   * @type {undefined|Uint32Array}
   * @private
   */
  this._indexes = undefined;
  /**
   * The number of splats in the primitive.
   * This is the total number of splats across all selected tiles.
   * @type {number}
   * @private
   */
  this._numSplats = 0;
  /**
   * Indicates whether or not the primitive needs a Gaussian splat texture.
   * This is set to true when the primitive is first created or when the splat attributes change.
   * @type {boolean}
   * @private
   */
  this._needsGaussianSplatTexture = true;

  /**
   * The previous view matrix used to determine if the primitive needs to be updated.
   * This is used to avoid unnecessary updates when the view matrix hasn't changed.
   * @type {Matrix4}
   * @private
   */
  this._prevViewMatrix = new Matrix4();

  /**
   * Indicates whether or not to show the bounding volume of the primitive for debugging purposes.
   * This is used to visualize the bounding volume of the primitive in the scene.
   * @type {boolean}
   * @private
   */
  this._debugShowBoundingVolume = options.debugShowBoundingVolume ?? false;

  /**
   * The texture used to store the Gaussian splat attributes.
   * This texture is created from the splat attributes (positions, scales, rotations, colors)
   * and is used in the vertex shader to render the splats.
   * @type {undefined|Texture}
   * @private
   * @see {@link GaussianSplatTextureGenerator}
   */
  this.gaussianSplatTexture = undefined;

  /**
   * The texture used to store the spherical harmonics coefficients for the Gaussian splats.
   * @type {undefined|Texture}
   * @private
   */
  this.sphericalHarmonicsTexture = undefined;

  /**
   * The last width of the Gaussian splat texture.
   * This is used to track changes in the texture size and update the primitive accordingly.
   * @type {number}
   * @private
   */
  this._lastTextureWidth = 0;
  /**
   * The last height of the Gaussian splat texture.
   * This is used to track changes in the texture size and update the primitive accordingly.
   * @type {number}
   * @private
   */
  this._lastTextureHeight = 0;

  /**
   * The pick ID used for color-buffer picking.
   * @type {undefined|PickId}
   * @private
   */
  this._pickId = undefined;

  /**
   * The vertex array used to render the Gaussian splats.
   * This vertex array contains the attributes needed to render the splats, such as positions and indexes.
   * @type {undefined|VertexArray}
   * @private
   */
  this._vertexArray = undefined;
  /**
   * The length of the vertex array, used to track changes in the number of splats.
   * This is used to determine if the vertex array needs to be rebuilt.
   * @type {number}
   * @private
   */
  this._vertexArrayLen = -1;
  this._splitDirection = SplitDirection.NONE;

  /**
   * The dirty flag forces the primitive to render this frame.
   * @type {boolean}
   * @private
   */
  this._dirty = false;

  this._tileset = options.tileset;

  this._baseTilesetUpdate = this._tileset.update;
  this._tileset.update = this._wrappedUpdate.bind(this);

  this._tileset.tileLoad.addEventListener(this.onTileLoad, this);
  this._tileset.tileVisible.addEventListener(this.onTileVisible, this);

  /**
   * Tracks current count of selected tiles.
   * This is used to determine if the primitive needs to be rebuilt.
   * @type {number}
   * @private
   */
  this.selectedTileLength = 0;

  /**
   * Indicates whether or not the primitive is ready for use.
   * @type {boolean}
   * @private
   */
  this._ready = false;

  /**
   * Indicates whether or not the primitive has a Gaussian splat texture.
   * @type {boolean}
   * @private
   */
  this._hasGaussianSplatTexture = false;

  /**
   * Indicates whether or not the primitive is currently generating a Gaussian splat texture.
   * @type {boolean}
   * @private
   */
  this._gaussianSplatTexturePending = false;

  /**
   * The draw command used to render the Gaussian splats.
   * @type {undefined|DrawCommand}
   * @private
   */
  this._drawCommand = undefined;
  /**
   * The root transform of the tileset.
   * This is used to transform the splats into world space.
   * @type {undefined|Matrix4}
   * @private
   */
  this._rootTransform = undefined;

  /**
   * The axis correction matrix to transform the splats from Y-up to Z-up.
   * @type {Matrix4}
   * @private
   */
  this._axisCorrectionMatrix = ModelUtility.getAxisCorrectionMatrix(
    Axis.Y,
    Axis.X,
    new Matrix4(),
  );

  /**
   * Indicates whether or not the primitive has been destroyed.
   * @type {boolean}
   * @private
   */
  this._isDestroyed = false;

  /**
   * The state of the Gaussian splat sorting process.
   * This is used to track the progress of the sorting operation.
   * @type {GaussianSplatSortingState}
   * @private
   */
  this._sorterState = GaussianSplatSortingState.IDLE;
  /**
   * A promise that resolves when the Gaussian splat sorting operation is complete.
   * This is used to track the progress of the sorting operation.
   * @type {undefined|Promise}
   * @private
   */
  this._sorterPromise = undefined;

  /**
   * An error that occurred during the Gaussian splat sorting operation.
   * Thrown when state is ERROR.
   * @type {undefined|Error}
   * @private
   */
  this._sorterError = undefined;

  /**
   * PLY index mapping data for tracking original PLY point indices.
   * @type {Array<Object>}
   * @private
   */
  this._tilePlyIndexOffsets = [];

  /**
   * Map from PLY index to aggregate index.
   * @type {Map<number, number>}
   * @private
   */
  this._plyIndexToAggregateIndex = new Map();

  /**
   * Color modifications by PLY index.
   * @type {Map<number, Array<number>>}
   * @private
   */
  this._colorModifications = new Map();

  /**
   * Custom color filters supplied through the public Cesium3DTileset API.
   * Each entry is an object with: { id, name, callback, enabled, priority, options, userData }.
   * @type {Array<Object>}
   * @private
   */
  this._customColorFilters = [];

  /**
   * The texture used to store the selection state of each Gaussian splat.
   * Each splat's state is stored as a single byte: bit 0 = selected, bit 1 = locked, bit 2 = deleted.
   * @type {undefined|Texture}
   * @private
   */
  this._splatStateTexture = undefined;

  /**
   * The state data for each splat (Uint8Array).
   * State encoding:
   *   - Bits 0-1: selected (bit 0) and locked (bit 1) flags
   *   - Bits 2-7: color group ID (0-63, 0 = default selection color)
   * @type {undefined|Uint8Array}
   * @private
   */
  this._splatStates = undefined;

  /**
   * Indicates whether the state texture needs to be updated.
   * @type {boolean}
   * @private
   */
  this._needsStateTextureUpdate = false;

  /**
   * The selected color for highlighting selected splats.
   * @type {Color}
   * @private
   */
  this._selectedColor = new Color(1.0, 0.0, 1.0, 0.5); // Magenta with 50% opacity

  /**
   * The locked color for highlighting locked splats.
   * @type {Color}
   * @private
   */
  this._lockedColor = new Color(0.5, 0.5, 0.5, 0.3); // Gray with 30% opacity

  /**
   * Set of selected splat indices (aggregate indices).
   * @type {Set<number>}
   * @private
   */
  this._selectedSplatIndices = new Set();

  /**
   * Set of locked splat indices (aggregate indices).
   * @type {Set<number>}
   * @private
   */
  this._lockedSplatIndices = new Set();

  /**
   * Color groups for multi-color highlighting.
   * Each group has a unique color. Group ID 0 is reserved for default selection color.
   * @type {Array<Color>}
   * @private
   */
  this._colorGroups = [];

  /**
   * Map from splat aggregate index to color group ID.
   * @type {Map<number, number>}
   * @private
   */
  this._splatColorGroups = new Map();

  /**
   * Maximum number of color groups supported (0-63, using 6 bits).
   * @type {number}
   * @private
   */
  this._maxColorGroups = 64;

  /**
   * Whether outline selection highlighting is enabled.
   * @type {boolean}
   * @private
   */
  this._outlineEnabled = false;

  /**
   * Outline color used for selection borders.
   * @type {Color}
   * @private
   */
  this._outlineColor = new Color(1.0, 0.5, 0.0, 1.0);

  /**
   * Outline width multiplier (in pixels).
   * @type {number}
   * @private
   */
  this._outlineWidth = 1.5;

  /**
   * Threshold applied when sampling outline mask alpha.
   * Default: 0.4 (centers mode, similar to Supersplat)
   * @type {number}
   * @private
   */
  this._outlineAlphaCutoff = 0.4; // 默认使用 centers 模式阈值（参考 Supersplat）

  /**
   * Outline rendering mode: 'centers' or 'rings'
   * 'centers': Uses Gaussian alpha with higher threshold (0.4) - thinner outline
   * 'rings': Uses solid alpha with lower threshold (0.0) - thicker, more continuous outline
   * @type {string}
   * @private
   */
  this._outlineMode = "centers";

  /**
   * Kernel radius (in pixels) for outline detection.
   * @type {number}
   * @private
   */
  this._outlineKernelRadius = 2;

  /**
   * Minimum alpha difference required to classify an edge (post-mask).
   * Separating该值可以避免阈值互相抵消。
   * @type {number}
   * @private
   */
  this._outlineMinAlphaDiff = 0.5;

  /**
   * Framebuffer used to render outline mask.
   * @type {FramebufferManager}
   * @private
   */
  this._outlineFramebuffer = undefined;

  /**
   * Texture storing the outline mask results.
   * @type {Texture}
   * @private
   */
  this._outlineMaskTexture = undefined;

  /**
   * Depth texture captured during outline mask rendering (for occlusion tests).
   * @type {Texture}
   * @private
   */
  this._outlineDepthTexture = undefined;

  /**
   * Cached texel step for outline post-process.
   * @type {Cartesian2}
   * @private
   */
  this._outlineTexelStep = new Cartesian2();

  /**
   * Pixel ratio captured for outline width scaling.
   * @type {number}
   * @private
   */
  this._outlinePixelRatio = 1.0;

  /**
   * Draw command used to render the outline mask.
   * @type {DrawCommand}
   * @private
   */
  this._outlineMaskCommand = undefined;

  /**
   * Viewport quad command used to composite the outline effect.
   * @type {DrawCommand}
   * @private
   */
  this._outlineCompositeCommand = undefined;

  /**
   * Clear command for the outline framebuffer.
   * @type {ClearCommand}
   * @private
   */
  this._outlineClearCommand = undefined;

  /**
   * Flag to track if outline parameters have changed and framebuffer needs clearing.
   * @type {boolean}
   * @private
   */
  this._outlineParamsChanged = false;

  /**
   * Enable debug logging for outline detection.
   * @type {boolean}
   * @private
   */
  this._outlineDebugEnabled = false;

  /**
   * Last debug parameters to avoid duplicate logs.
   * @type {Object}
   * @private
   */
  this._outlineDebugLastParams = undefined;

  /**
   * Enable debug visualization mode (color-coded edge detection).
   * @type {boolean}
   * @private
   */
  this._outlineDebugMode = false;

  /**
   * Whether depth testing is applied during outline composite.
   * @type {boolean}
   * @private
   */
  this._outlineDepthTestEnabled = true;

  /**
   * Whether boundary cache needs update for ring outlines.
   * @type {boolean}
   * @private
   */
  this._boundaryDirty = true;

  /**
   * Cached boundary splat indices (aggregate indices).
   * @type {Uint32Array|undefined}
   * @private
   */
  this._boundaryIndices = undefined;

  /**
   * Vertex array for boundary ring rendering.
   * @type {VertexArray|undefined}
   * @private
   */
  this._boundaryVertexArray = undefined;

  /**
   * Instance count for boundary ring rendering.
   * @type {number}
   * @private
   */
  this._boundaryInstanceCount = 0;

  /**
   * DrawCommand for boundary ring rendering.
   * @type {DrawCommand|undefined}
   * @private
   */
  this._outlineRingCommand = undefined;

  /**
   * Cell size multiplier used when clustering selected splats for boundary detection.
   * @type {number}
   * @private
   */
  this._outlineBoundaryCellMultiplier = 2.5;

  /**
   * Minimum cell size in meters for boundary clustering.
   * @type {number}
   * @private
   */
  this._outlineBoundaryMinCellSize = 0.5;

  /**
   * Neighbor search range (in grid cells) used to determine if a cell is at boundary.
   * @type {number}
   * @private
   */
  this._outlineBoundaryNeighborRange = 1;

  /**
   * Minimum neighbor coverage ratio (0-1) to treat a cell as interior.
   * Cells below this ratio are considered boundary.
   * @type {number}
   * @private
   */
  this._outlineBoundaryMinCoverage = 0.75;

  /**
   * Cached statistics for the last boundary computation.
   * @type {Object|undefined}
   * @private
   */
  this._outlineBoundaryStats = undefined;
}

Object.defineProperties(GaussianSplatPrimitive.prototype, {
  /**
   * Indicates whether the primitive is ready for use.
   * @memberof GaussianSplatPrimitive.prototype
   * @type {boolean}
   * @readonly
   */
  ready: {
    get: function () {
      return this._ready;
    },
  },

  /**
   * The {@link SplitDirection} to apply to this point.
   * @memberof GaussianSplatPrimitive.prototype
   * @type {SplitDirection}
   * @default {@link SplitDirection.NONE}
   */
  splitDirection: {
    get: function () {
      return this._splitDirection;
    },
    set: function (value) {
      if (this._splitDirection !== value) {
        this._splitDirection = value;
        this._dirty = true;
      }
    },
  },

  /**
   * The selected color for highlighting selected splats.
   * @memberof GaussianSplatPrimitive.prototype
   * @type {Color}
   */
  selectedColor: {
    get: function () {
      return this._selectedColor;
    },
    set: function (value) {
      if (defined(value)) {
        Color.clone(value, this._selectedColor);
        this._dirty = true;
      }
    },
  },

  /**
   * The locked color for highlighting locked splats.
   * @memberof GaussianSplatPrimitive.prototype
   * @type {Color}
   */
  lockedColor: {
    get: function () {
      return this._lockedColor;
    },
    set: function (value) {
      if (defined(value)) {
        Color.clone(value, this._lockedColor);
        this._dirty = true;
      }
    },
  },
});

/**
 * Sets the selection state of splats by their aggregate indices.
 * @param {Array<number>|Set<number>} indices The aggregate indices of splats to select.
 * @param {boolean} [selected=true] Whether to select (true) or deselect (false) the splats.
 */
GaussianSplatPrimitive.prototype.setSplatSelection = function (
  indices,
  selected,
) {
  selected = selected !== false; // default to true
  const indicesArray = Array.isArray(indices) ? indices : Array.from(indices);

  if (!this._selectionLogShown) {
    console.log("[选中高亮] setSplatSelection 调用:");
    console.log("  - 总 splat 数:", this._numSplats);
    console.log("  - 输入索引:", indicesArray);
    console.log("  - 操作:", selected ? "选中" : "取消选中");
    this._selectionLogShown = true;
  }

  let validCount = 0;
  let invalidCount = 0;
  let maxIndex = -1;
  let minIndex = Infinity;

  for (let i = 0; i < indicesArray.length; i++) {
    const index = indicesArray[i];
    if (index >= 0 && index < this._numSplats) {
      if (selected) {
        this._selectedSplatIndices.add(index);
      } else {
        this._selectedSplatIndices.delete(index);
      }
      validCount++;
      if (index > maxIndex) {
        maxIndex = index;
      }
      if (index < minIndex) {
        minIndex = index;
      }
    } else {
      invalidCount++;
      if (invalidCount <= 5) {
        console.warn(
          `[选中高亮] ⚠️ 无效索引: ${index} (范围: 0-${this._numSplats - 1})`,
        );
      }
    }
  }

  console.log(
    `[选中高亮] 已${selected ? "选中" : "取消选中"} ${validCount} 个 splats`,
  );
  if (invalidCount > 0) {
    console.warn(`[选中高亮] ⚠️ ${invalidCount} 个无效索引被忽略`);
    console.warn(`[选中高亮] 提示: 索引范围应为 0-${this._numSplats - 1}`);
    console.warn(
      `[选中高亮] 如果输入的是 PLY 索引，需要使用 PLY 索引到聚合索引的映射`,
    );
  }
  if (validCount > 0) {
    console.log(`[选中高亮] 有效索引范围: ${minIndex} - ${maxIndex}`);
  }
  console.log(`[选中高亮] 当前选中数量: ${this._selectedSplatIndices.size}`);

  this._needsStateTextureUpdate = true;
  this._dirty = true;
  this._markBoundaryDirty();
};

/**
 * Sets the lock state of splats by their aggregate indices.
 * @param {Array<number>|Set<number>} indices The aggregate indices of splats to lock.
 * @param {boolean} [locked=true] Whether to lock (true) or unlock (false) the splats.
 */
GaussianSplatPrimitive.prototype.setSplatLock = function (indices, locked) {
  locked = locked !== false; // default to true
  const indicesArray = Array.isArray(indices) ? indices : Array.from(indices);

  if (!this._lockLogShown) {
    console.log("[锁定高亮] setSplatLock 调用:");
    console.log("  - 总 splat 数:", this._numSplats);
    console.log("  - 输入索引:", indicesArray);
    console.log("  - 操作:", locked ? "锁定" : "解锁");
    this._lockLogShown = true;
  }

  let validCount = 0;
  let invalidCount = 0;

  for (let i = 0; i < indicesArray.length; i++) {
    const index = indicesArray[i];
    if (index >= 0 && index < this._numSplats) {
      if (locked) {
        this._lockedSplatIndices.add(index);
      } else {
        this._lockedSplatIndices.delete(index);
      }
      validCount++;
    } else {
      invalidCount++;
      if (invalidCount <= 5) {
        console.warn(
          `[锁定高亮] ⚠️ 无效索引: ${index} (范围: 0-${this._numSplats - 1})`,
        );
      }
    }
  }

  console.log(
    `[锁定高亮] 已${locked ? "锁定" : "解锁"} ${validCount} 个 splats`,
  );
  if (invalidCount > 0) {
    console.warn(`[锁定高亮] ⚠️ ${invalidCount} 个无效索引被忽略`);
  }
  console.log(`[锁定高亮] 当前锁定数量: ${this._lockedSplatIndices.size}`);

  this._needsStateTextureUpdate = true;
  this._dirty = true;
};

/**
 * Clears all selection states.
 */
GaussianSplatPrimitive.prototype.clearSelection = function () {
  this._selectedSplatIndices.clear();
  this._needsStateTextureUpdate = true;
  this._dirty = true;
  this._markBoundaryDirty();
};

/**
 * Clears all lock states.
 */
GaussianSplatPrimitive.prototype.clearLocks = function () {
  this._lockedSplatIndices.clear();
  this._needsStateTextureUpdate = true;
  this._dirty = true;
};

GaussianSplatPrimitive.prototype._markBoundaryDirty = function () {
  this._boundaryDirty = true;
  this._boundaryIndices = undefined;
  this._boundaryInstanceCount = 0;
  if (defined(this._boundaryVertexArray)) {
    this._boundaryVertexArray.destroy();
    this._boundaryVertexArray = undefined;
  }
  this._outlineRingCommand = undefined;
  this._outlineBoundaryStats = undefined;
};

/**
 * Computes the nearest splat to the provided window position and returns metadata such as color group ID and PLY index.
 * Only currently selected splats are considered to avoid scanning the entire dataset.
 *
 * @param {Scene} scene The scene used for coordinate transforms.
 * @param {Cartesian2} windowPosition The window position in pixels.
 * @param {object} [options] Additional options.
 * @param {number} [options.maxDistance=40.0] Maximum allowed pixel distance in drawing-buffer space.
 * @param {boolean} [options.searchSelectedOnly=true] When true, restricts the search to currently selected splats.
 * @returns {object|undefined} Metadata object if a splat is found, otherwise <code>undefined</code>.
 */
GaussianSplatPrimitive.prototype.getSplatInfoAtScreenPosition = function (
  scene,
  windowPosition,
  options,
) {
  //>>includeStart('debug', pragmas.debug);
  console.log(
    "[拾取流程] GaussianSplatPrimitive.getSplatInfoAtScreenPosition() 被调用",
  );
  console.log("  - windowPosition:", windowPosition);
  console.log("  - options:", options);
  //>>includeEnd('debug');

  if (!defined(scene) || !defined(windowPosition)) {
    //>>includeStart('debug', pragmas.debug);
    console.log(
      "[拾取流程] getSplatInfoAtScreenPosition() 参数无效，返回 undefined",
    );
    //>>includeEnd('debug');
    return undefined;
  }

  if (!defined(this._positions) || this._positions.length === 0) {
    //>>includeStart('debug', pragmas.debug);
    console.log(
      "[拾取流程] getSplatInfoAtScreenPosition() 位置数据不存在，返回 undefined",
    );
    //>>includeEnd('debug');
    return undefined;
  }

  const searchSelectedOnly =
    !defined(options) || !defined(options.searchSelectedOnly)
      ? true
      : options.searchSelectedOnly;
  const maxDistance =
    defined(options) && defined(options.maxDistance)
      ? options.maxDistance
      : 40.0;
  const worldMaxDistance =
    defined(options) && defined(options.worldMaxDistance)
      ? options.worldMaxDistance
      : 0.05;

  //>>includeStart('debug', pragmas.debug);
  console.log("[拾取流程] getSplatInfoAtScreenPosition() 搜索参数:");
  console.log("  - searchSelectedOnly:", searchSelectedOnly);
  console.log("  - maxDistance:", maxDistance);
  console.log("  - worldMaxDistance:", worldMaxDistance);
  console.log("  - 已选中 splat 数量:", this._selectedSplatIndices.size);
  //>>includeEnd('debug');

  let candidateIndices;
  if (searchSelectedOnly && this._selectedSplatIndices.size > 0) {
    candidateIndices = Array.from(this._selectedSplatIndices);
    //>>includeStart('debug', pragmas.debug);
    console.log(
      "[拾取流程] 使用已选中的 splats 作为候选:",
      candidateIndices.length,
    );
    //>>includeEnd('debug');
  } else if (
    defined(options) &&
    defined(options.indices) &&
    options.indices.length > 0
  ) {
    candidateIndices = options.indices;
    //>>includeStart('debug', pragmas.debug);
    console.log(
      "[拾取流程] 使用提供的 indices 作为候选:",
      candidateIndices.length,
    );
    //>>includeEnd('debug');
  } else if (!searchSelectedOnly) {
    // fallback to entire dataset (may be large) – skip for now to avoid huge scans
    candidateIndices = undefined;
    //>>includeStart('debug', pragmas.debug);
    console.log("[拾取流程] 警告: 未限制搜索范围，跳过整个数据集扫描");
    //>>includeEnd('debug');
  }

  if (!defined(candidateIndices) || candidateIndices.length === 0) {
    //>>includeStart('debug', pragmas.debug);
    console.log(
      "[拾取流程] getSplatInfoAtScreenPosition() 无候选索引，返回 undefined",
    );
    //>>includeEnd('debug');
    return undefined;
  }

  const drawingBufferPosition = SceneTransforms.transformWindowToDrawingBuffer(
    scene,
    windowPosition,
    scratchPickDrawingBuffer,
  );
  if (!defined(drawingBufferPosition)) {
    //>>includeStart('debug', pragmas.debug);
    console.log(
      "[拾取流程] getSplatInfoAtScreenPosition() drawingBufferPosition 转换失败，返回 undefined",
    );
    //>>includeEnd('debug');
    return undefined;
  }

  const pickWorldSupported = scene.pickPositionSupported === true;
  const pickWorldPosition =
    pickWorldSupported && defined(scene.pickPosition)
      ? scene.pickPosition(windowPosition, scratchPickWorldPosition)
      : undefined;

  //>>includeStart('debug', pragmas.debug);
  console.log("[拾取流程] getSplatInfoAtScreenPosition() 坐标转换结果:");
  console.log("  - drawingBufferPosition:", drawingBufferPosition);
  console.log("  - pickWorldSupported:", pickWorldSupported);
  console.log("  - pickWorldPosition:", pickWorldPosition ? "存在" : "不存在");
  if (pickWorldPosition) {
    console.log("  - pickWorldPosition 值:", pickWorldPosition);
  }
  // 检查 _positions 的第一个 splat 位置作为参考
  if (defined(this._positions) && this._positions.length >= 3) {
    console.log("  - 第一个 splat 位置 (参考):", {
      x: this._positions[0].toFixed(2),
      y: this._positions[1].toFixed(2),
      z: this._positions[2].toFixed(2),
    });
    if (pickWorldPosition) {
      const firstSplatPos = new Cartesian3(
        this._positions[0],
        this._positions[1],
        this._positions[2],
      );
      const distanceToFirst = Cartesian3.distance(
        pickWorldPosition,
        firstSplatPos,
      );
      console.log(
        "  - 点击位置到第一个 splat 的距离:",
        distanceToFirst.toFixed(2),
      );
    }
  }
  //>>includeEnd('debug');

  let closestIndex = -1;
  let closestDistance = maxDistance;
  let closestWorldIndex = -1;
  let closestWorldDistance = worldMaxDistance;

  // Transform positions from root space to world space
  // _positions are stored in root space (relative to _rootTransform)
  // We need to transform them to world space for distance calculation
  const tileset = this._tileset;
  let rootToWorldMatrix;
  if (defined(this._rootTransform) && defined(tileset.modelMatrix)) {
    rootToWorldMatrix = Matrix4.multiply(
      tileset.modelMatrix,
      this._rootTransform,
      scratchMatrix4A,
    );
  } else {
    // Fallback: assume positions are already in world space
    rootToWorldMatrix = Matrix4.IDENTITY;
  }

  //>>includeStart('debug', pragmas.debug);
  console.log("[拾取流程] 坐标转换矩阵:");
  console.log("  - _rootTransform 存在:", defined(this._rootTransform));
  console.log("  - tileset.modelMatrix 存在:", defined(tileset?.modelMatrix));
  console.log("  - rootToWorldMatrix 存在:", defined(rootToWorldMatrix));
  if (defined(this._rootTransform) && defined(tileset.modelMatrix)) {
    console.log("  - 使用 root space → world space 转换");
  } else {
    console.log("  - 警告: 假设 positions 已在世界空间");
  }
  //>>includeEnd('debug');

  let validProjectionCount = 0;
  let invalidProjectionCount = 0;
  let outOfRangeCount = 0;
  let distanceTooFarCount = 0;
  let worldDistanceTooFarCount = 0;
  let minPixelDistance = Infinity;
  let minWorldDistance = Infinity;

  for (let i = 0; i < candidateIndices.length; i++) {
    const index = candidateIndices[i];
    const base = index * 3;
    if (base + 2 >= this._positions.length) {
      outOfRangeCount++;
      continue;
    }

    // Get position in root space
    scratchPickCartesian.x = this._positions[base];
    scratchPickCartesian.y = this._positions[base + 1];
    scratchPickCartesian.z = this._positions[base + 2];

    // Transform from root space to world space
    const worldPosition = Matrix4.multiplyByPoint(
      rootToWorldMatrix,
      scratchPickCartesian,
      scratchPickWorldPosition,
    );

    //>>includeStart('debug', pragmas.debug);
    // 记录前几个 splat 的位置用于调试
    if (i < 3) {
      console.log(`[拾取流程] 候选 splat ${i} (索引 ${index}):`);
      console.log(`  - Root space 坐标:`, {
        x: scratchPickCartesian.x.toFixed(2),
        y: scratchPickCartesian.y.toFixed(2),
        z: scratchPickCartesian.z.toFixed(2),
      });
      console.log(`  - World space 坐标:`, {
        x: worldPosition.x.toFixed(2),
        y: worldPosition.y.toFixed(2),
        z: worldPosition.z.toFixed(2),
      });
    }
    //>>includeEnd('debug');

    const projected = SceneTransforms.worldToDrawingBufferCoordinates(
      scene,
      worldPosition,
      scratchPickProjected,
    );
    if (!defined(projected)) {
      invalidProjectionCount++;
      continue;
    }

    validProjectionCount++;

    const dx = projected.x - drawingBufferPosition.x;
    const dy = projected.y - drawingBufferPosition.y;
    const distance = Math.sqrt(dx * dx + dy * dy);

    if (distance < minPixelDistance) {
      minPixelDistance = distance;
    }

    if (distance <= closestDistance) {
      closestDistance = distance;
      closestIndex = index;
    } else {
      distanceTooFarCount++;
    }

    if (defined(pickWorldPosition)) {
      // Use world space position for distance calculation
      const worldDistance = Cartesian3.distance(
        pickWorldPosition,
        worldPosition,
      );

      if (worldDistance < minWorldDistance) {
        minWorldDistance = worldDistance;
      }

      if (worldDistance <= closestWorldDistance) {
        closestWorldDistance = worldDistance;
        closestWorldIndex = index;
      } else {
        worldDistanceTooFarCount++;
      }
    }
  }

  //>>includeStart('debug', pragmas.debug);
  console.log("[拾取流程] getSplatInfoAtScreenPosition() 搜索统计:");
  console.log("  - 候选索引总数:", candidateIndices.length);
  console.log("  - 有效投影数:", validProjectionCount);
  console.log("  - 无效投影数:", invalidProjectionCount);
  console.log("  - 超出范围数:", outOfRangeCount);
  console.log("  - 像素距离过远数:", distanceTooFarCount);
  console.log("  - 世界距离过远数:", worldDistanceTooFarCount);
  console.log(
    "  - 最小像素距离:",
    minPixelDistance === Infinity ? "N/A" : minPixelDistance.toFixed(2),
  );
  console.log(
    "  - 最小世界距离:",
    minWorldDistance === Infinity ? "N/A" : minWorldDistance.toFixed(6),
  );
  console.log("  - 最大允许像素距离:", maxDistance);
  console.log("  - 最大允许世界距离:", worldMaxDistance);
  console.log("  - closestIndex:", closestIndex);
  console.log("  - closestWorldIndex:", closestWorldIndex);
  if (closestIndex !== -1) {
    console.log("  - closestDistance:", closestDistance.toFixed(2));
  }
  if (closestWorldIndex !== -1) {
    console.log("  - closestWorldDistance:", closestWorldDistance.toFixed(6));
  }
  //>>includeEnd('debug');

  let finalIndex = closestIndex;
  if (finalIndex === -1 && closestWorldIndex !== -1) {
    finalIndex = closestWorldIndex;
    closestDistance = closestWorldDistance;
    //>>includeStart('debug', pragmas.debug);
    console.log("[拾取流程] 使用世界距离找到 splat，索引:", finalIndex);
    //>>includeEnd('debug');
  }

  if (finalIndex === -1) {
    //>>includeStart('debug', pragmas.debug);
    console.log(
      "[拾取流程] getSplatInfoAtScreenPosition() 未找到匹配的 splat，返回 undefined",
    );
    console.log("  可能原因分析:");
    if (validProjectionCount === 0) {
      console.log("    ❌ 所有候选 splat 的投影坐标都无效");
      console.log("       - 可能 splat 位置在视锥体外");
      console.log("       - 可能 splat 位置被遮挡");
    } else if (
      minPixelDistance > maxDistance &&
      minWorldDistance > worldMaxDistance
    ) {
      console.log("    ❌ 所有候选 splat 的距离都超过阈值");
      console.log(
        "       - 最小像素距离:",
        minPixelDistance.toFixed(2),
        ">",
        maxDistance,
      );
      console.log(
        "       - 最小世界距离:",
        minWorldDistance.toFixed(6),
        ">",
        worldMaxDistance,
      );
      console.log("       - 建议: 增大 maxDistance 或 worldMaxDistance");
    } else if (minPixelDistance > maxDistance) {
      console.log("    ❌ 所有候选 splat 的像素距离都超过阈值");
      console.log(
        "       - 最小像素距离:",
        minPixelDistance.toFixed(2),
        ">",
        maxDistance,
      );
      console.log("       - 建议: 增大 maxDistance (当前:", maxDistance, ")");
    } else if (minWorldDistance > worldMaxDistance) {
      console.log("    ❌ 所有候选 splat 的世界距离都超过阈值");
      console.log(
        "       - 最小世界距离:",
        minWorldDistance.toFixed(6),
        ">",
        worldMaxDistance,
      );
      console.log(
        "       - 建议: 增大 worldMaxDistance (当前:",
        worldMaxDistance,
        ")",
      );
    } else {
      console.log("    ❌ 未知原因，请检查代码逻辑");
    }
    //>>includeEnd('debug');
    return undefined;
  }

  const plyIndex = defined(this._plyIndicesAggregate)
    ? this._plyIndicesAggregate[finalIndex]
    : undefined;
  const colorGroupId = this._splatColorGroups.get(finalIndex) ?? 0;

  const result = {
    aggregateIndex: finalIndex,
    plyIndex: plyIndex,
    colorGroupId: colorGroupId,
    isSelected: this._selectedSplatIndices.has(finalIndex),
    isLocked: this._lockedSplatIndices.has(finalIndex),
    pixelDistance: closestDistance,
  };

  //>>includeStart('debug', pragmas.debug);
  console.log("[拾取流程] getSplatInfoAtScreenPosition() 找到 splat 信息:");
  console.log("  - aggregateIndex:", result.aggregateIndex);
  console.log("  - plyIndex:", result.plyIndex);
  console.log("  - colorGroupId:", result.colorGroupId);
  console.log("  - isSelected:", result.isSelected);
  console.log("  - isLocked:", result.isLocked);
  console.log("  - pixelDistance:", result.pixelDistance);
  //>>includeEnd('debug');

  return result;
};

/**
 * Sets the color group for splats by their aggregate indices.
 * @param {Array<number>|Set<number>} indices The aggregate indices of splats.
 * @param {number} [groupId=0] The color group ID (0-63). 0 uses default selection color.
 */
GaussianSplatPrimitive.prototype.setSplatColorGroup = function (
  indices,
  groupId,
) {
  groupId = groupId !== undefined ? groupId : 0;
  const indicesArray = Array.isArray(indices) ? indices : Array.from(indices);

  // Clamp group ID to valid range
  const clampedGroupId = Math.min(
    Math.max(0, groupId),
    this._maxColorGroups - 1,
  );

  if (!this._colorGroupLogShown) {
    console.log("[颜色组] setSplatColorGroup 调用:");
    console.log("  - 总 splat 数:", this._numSplats);
    console.log("  - 输入索引数量:", indicesArray.length);
    console.log("  - 颜色组ID:", clampedGroupId);
    this._colorGroupLogShown = true;
  }

  let validCount = 0;
  let invalidCount = 0;

  for (let i = 0; i < indicesArray.length; i++) {
    const index = indicesArray[i];
    if (index >= 0 && index < this._numSplats) {
      if (clampedGroupId === 0) {
        // Remove color group assignment (use default)
        this._splatColorGroups.delete(index);
      } else {
        this._splatColorGroups.set(index, clampedGroupId);
      }
      validCount++;
    } else {
      invalidCount++;
      if (invalidCount <= 5) {
        console.warn(
          `[颜色组] ⚠️ 无效索引: ${index} (范围: 0-${this._numSplats - 1})`,
        );
      }
    }
  }

  console.log(
    `[颜色组] 已为 ${validCount} 个 splats 设置颜色组 ${clampedGroupId}`,
  );
  if (invalidCount > 0) {
    console.warn(`[颜色组] ⚠️ ${invalidCount} 个无效索引被忽略`);
  }

  this._needsStateTextureUpdate = true;
  this._dirty = true;
};

/**
 * Sets the color for a color group.
 * @param {number} groupId The color group ID (1-63, 0 is reserved for default).
 * @param {Color} color The color for this group.
 */
GaussianSplatPrimitive.prototype.setColorGroupColor = function (
  groupId,
  color,
) {
  if (groupId < 1 || groupId >= this._maxColorGroups) {
    console.warn(
      `[颜色组] ⚠️ 无效的颜色组ID: ${groupId} (范围: 1-${this._maxColorGroups - 1})`,
    );
    return;
  }

  // Ensure color groups array is large enough
  while (this._colorGroups.length <= groupId) {
    this._colorGroups.push(new Color(1.0, 0.0, 1.0, 0.5)); // Default magenta
  }

  this._colorGroups[groupId] = Color.clone(color);
  console.log(
    `[颜色组] 颜色组 ${groupId} 颜色已设置为:`,
    color.toCssColorString(),
  );

  this._dirty = true;
};

/**
 * Gets the color for a color group.
 * @param {number} groupId The color group ID.
 * @returns {Color|undefined} The color for this group, or undefined if not set.
 */
GaussianSplatPrimitive.prototype.getColorGroupColor = function (groupId) {
  if (groupId < 0 || groupId >= this._maxColorGroups) {
    return undefined;
  }
  if (groupId === 0) {
    return this._selectedColor; // Default selection color
  }
  return this._colorGroups[groupId];
};

/**
 * Clears all color group assignments (splats will use default selection color).
 */
GaussianSplatPrimitive.prototype.clearColorGroups = function () {
  this._splatColorGroups.clear();
  this._needsStateTextureUpdate = true;
  this._dirty = true;
  console.log("[颜色组] 已清除所有颜色组分配");
};

/**
 * Enables or disables outline selection highlighting.
 * @param {boolean} enabled
 */
GaussianSplatPrimitive.prototype.setOutlineSelectionEnabled = function (
  enabled,
) {
  const value = enabled === true;
  if (this._outlineEnabled === value) {
    return;
  }
  this._outlineEnabled = value;
  if (!value) {
    destroyOutlineResources(this);
  }
  this._dirty = true;
};

/**
 * Sets the outline color.
 * @param {Color} color
 */
GaussianSplatPrimitive.prototype.setOutlineColor = function (color) {
  if (!defined(color)) {
    return;
  }
  Color.clone(color, this._outlineColor);
  this._outlineParamsChanged = true;
  this._dirty = true;
};

/**
 * Sets outline width multiplier.
 * @param {number} width Width in pixels (clamped to >= 0.5).
 */
GaussianSplatPrimitive.prototype.setOutlineWidth = function (width) {
  if (!defined(width)) {
    return;
  }
  this._outlineWidth = Math.max(0.1, width);
  this._outlineParamsChanged = true;
  this._dirty = true;
};

/**
 * Sets outline alpha cutoff used during mask sampling.
 * @param {number} cutoff Alpha threshold (0.0-1.0)
 * @param {string} [mode] Optional mode: 'centers' (default, threshold 0.4) or 'rings' (threshold 0.0)
 *                        If mode is provided, cutoff will be overridden by mode-specific value
 */
GaussianSplatPrimitive.prototype.setOutlineAlphaCutoff = function (
  cutoff,
  mode,
) {
  const oldCutoff = this._outlineAlphaCutoff;
  const oldMode = this._outlineMode;

  if (defined(mode)) {
    // 如果提供了模式，使用模式特定的阈值（参考 Supersplat）
    if (mode === "rings") {
      // Rings 模式：固定使用 0.0 阈值
      this._outlineAlphaCutoff = 0.0;
      this._outlineMode = "rings";
    } else if (mode === "centers") {
      // Centers 模式：使用用户提供的 cutoff 值，如果未提供则使用默认值 0.4
      if (defined(cutoff)) {
        this._outlineAlphaCutoff = CesiumMath.clamp(cutoff, 0.0, 1.0);
      } else {
        this._outlineAlphaCutoff = 0.4; // 默认值
      }
      this._outlineMode = "centers";
    } else {
      // 无效模式，使用提供的 cutoff 值
      this._outlineAlphaCutoff = defined(cutoff)
        ? CesiumMath.clamp(cutoff, 0.0, 1.0)
        : 0.4;
      this._outlineMode = "centers";
    }
  } else if (defined(cutoff)) {
    // 只提供 cutoff，手动设置（保持当前模式）
    this._outlineAlphaCutoff = CesiumMath.clamp(cutoff, 0.0, 1.0);
  } else {
    return;
  }

  // 调试日志
  if (this._outlineDebugEnabled) {
    console.log("[轮廓调试] setOutlineAlphaCutoff:", {
      oldCutoff: oldCutoff,
      newCutoff: this._outlineAlphaCutoff,
      oldMode: oldMode,
      newMode: this._outlineMode,
      providedCutoff: cutoff,
      providedMode: mode,
    });
  }

  this._outlineParamsChanged = true;
  this._dirty = true;
};

/**
 * Sets outline rendering mode.
 * @param {string} mode 'centers' or 'rings'
 *                      'centers': Thinner outline, uses alpha threshold 0.4
 *                      'rings': Thicker, more continuous outline, uses alpha threshold 0.0
 */
GaussianSplatPrimitive.prototype.setOutlineMode = function (mode) {
  if (!defined(mode)) {
    return;
  }
  const oldMode = this._outlineMode;
  const oldCutoff = this._outlineAlphaCutoff;

  if (mode === "rings") {
    // Rings 模式：固定使用 0.0 阈值
    this._outlineAlphaCutoff = 0.0;
    this._outlineMode = "rings";
  } else if (mode === "centers") {
    // Centers 模式：
    // - 如果从 rings 切换过来，使用默认值 0.4
    // - 如果已经是 centers 模式，保留当前阈值（允许用户自定义）
    if (oldMode === "rings") {
      this._outlineAlphaCutoff = 0.4; // 从 rings 切换，使用默认值
    } else {
      // 已经是 centers 模式，保留当前阈值（如果用户已经自定义过）
      // 如果当前是默认值 0.4，也保持不变
      // 这样用户可以先用 setOutlineAlphaCutoff 设置自定义值，再调用 setOutlineMode
    }
    this._outlineMode = "centers";
  } else {
    console.warn(
      `Invalid outline mode: ${mode}. Expected 'centers' or 'rings'.`,
    );
    return;
  }

  // 调试日志
  if (this._outlineDebugEnabled) {
    console.log("[轮廓调试] setOutlineMode:", {
      oldMode: oldMode,
      newMode: this._outlineMode,
      oldCutoff: oldCutoff,
      newCutoff: this._outlineAlphaCutoff,
      cutoffPreserved: mode === "centers" && oldMode === "centers",
    });
  }

  this._outlineParamsChanged = true;
  this._dirty = true;
};

/**
 * Sets the outline kernel radius (integer range 1-4).
 * @param {number} radius
 */
GaussianSplatPrimitive.prototype.setOutlineKernelRadius = function (radius) {
  if (!defined(radius)) {
    return;
  }
  const oldRadius = this._outlineKernelRadius;
  const clamped = CesiumMath.clamp(Math.round(radius), 1, 4);
  if (this._outlineKernelRadius === clamped) {
    return;
  }
  this._outlineKernelRadius = clamped;

  // 调试日志
  if (this._outlineDebugEnabled) {
    console.log("[轮廓调试] setOutlineKernelRadius:", {
      oldRadius: oldRadius,
      newRadius: this._outlineKernelRadius,
      providedRadius: radius,
      clamped: clamped,
    });
  }

  this._outlineParamsChanged = true;
  this._dirty = true;
};

/**
 * Sets the minimum alpha difference used to classify edges after mask sampling.
 * @param {number} diff Value in [0, 1].
 */
GaussianSplatPrimitive.prototype.setOutlineMinAlphaDifference = function (
  diff,
) {
  if (!defined(diff)) {
    return;
  }
  const clamped = CesiumMath.clamp(diff, 0.0, 1.0);
  if (this._outlineMinAlphaDiff === clamped) {
    return;
  }

  const oldDiff = this._outlineMinAlphaDiff;
  this._outlineMinAlphaDiff = clamped;

  if (this._outlineDebugEnabled) {
    console.log("[轮廓调试] setOutlineMinAlphaDifference:", {
      oldDiff: oldDiff,
      newDiff: this._outlineMinAlphaDiff,
    });
  }

  this._outlineParamsChanged = true;
  this._dirty = true;
};

/**
 * Enables or disables depth testing for the outline composite pass.
 * @param {boolean} enabled
 */
GaussianSplatPrimitive.prototype.setOutlineDepthTestEnabled = function (
  enabled,
) {
  const value = enabled !== false;
  if (this._outlineDepthTestEnabled === value) {
    return;
  }
  this._outlineDepthTestEnabled = value;

  if (this._outlineDebugEnabled) {
    console.log("[轮廓调试] setOutlineDepthTestEnabled:", {
      enabled: this._outlineDepthTestEnabled,
    });
  }

  this._outlineParamsChanged = true;
  this._dirty = true;
};

GaussianSplatPrimitive.prototype.setOutlineBoundaryParameters = function (
  options,
) {
  if (!defined(options)) {
    return;
  }

  const cellMultiplier = CesiumMath.clamp(
    defined(options.cellMultiplier)
      ? options.cellMultiplier
      : this._outlineBoundaryCellMultiplier,
    0.1,
    20.0,
  );
  const minCellSize = CesiumMath.clamp(
    defined(options.minCellSize)
      ? options.minCellSize
      : this._outlineBoundaryMinCellSize,
    0.01,
    10.0,
  );
  const neighborRange = CesiumMath.clamp(
    defined(options.neighborRange)
      ? Math.round(options.neighborRange)
      : this._outlineBoundaryNeighborRange,
    1,
    4,
  );
  const minCoverage = CesiumMath.clamp(
    defined(options.minCoverage)
      ? options.minCoverage
      : this._outlineBoundaryMinCoverage,
    0.0,
    1.0,
  );

  if (
    cellMultiplier === this._outlineBoundaryCellMultiplier &&
    minCellSize === this._outlineBoundaryMinCellSize &&
    neighborRange === this._outlineBoundaryNeighborRange &&
    minCoverage === this._outlineBoundaryMinCoverage
  ) {
    return;
  }

  this._outlineBoundaryCellMultiplier = cellMultiplier;
  this._outlineBoundaryMinCellSize = minCellSize;
  this._outlineBoundaryNeighborRange = neighborRange;
  this._outlineBoundaryMinCoverage = minCoverage;
  this._markBoundaryDirty();

  if (this._outlineDebugEnabled) {
    console.log("[轮廓调试] setOutlineBoundaryParameters:", {
      cellMultiplier: cellMultiplier,
      minCellSize: minCellSize,
      neighborRange: neighborRange,
      minCoverage: minCoverage,
    });
  }
};

GaussianSplatPrimitive.prototype.getOutlineBoundaryStats = function () {
  return defined(this._outlineBoundaryStats)
    ? clone(this._outlineBoundaryStats, true)
    : undefined;
};

/**
 * Enable or disable debug logging for outline detection.
 * @param {boolean} enabled
 */
GaussianSplatPrimitive.prototype.setOutlineDebugEnabled = function (enabled) {
  this._outlineDebugEnabled = enabled === true;
  if (this._outlineDebugEnabled) {
    console.log("[轮廓调试] 调试日志模式已启用");
    console.log("[轮廓调试] 当前轮廓参数:", {
      enabled: this._outlineEnabled,
      mode: this._outlineMode,
      alphaCutoff: this._outlineAlphaCutoff,
      minAlphaDiff: this._outlineMinAlphaDiff,
      kernelRadius: this._outlineKernelRadius,
      width: this._outlineWidth,
      depthTest: this._outlineDepthTestEnabled,
      boundary: {
        cellMultiplier: this._outlineBoundaryCellMultiplier,
        minCellSize: this._outlineBoundaryMinCellSize,
        neighborRange: this._outlineBoundaryNeighborRange,
        minCoverage: this._outlineBoundaryMinCoverage,
      },
      color: `rgba(${Math.round(this._outlineColor.red * 255)}, ${Math.round(this._outlineColor.green * 255)}, ${Math.round(this._outlineColor.blue * 255)}, ${this._outlineColor.alpha})`,
    });
  } else {
    console.log("[轮廓调试] 调试日志模式已禁用");
    this._outlineDebugLastParams = undefined;
  }
  this._dirty = true;
};

/**
 * Enable or disable debug visualization mode (color-coded edge detection).
 * When enabled, outline color will be replaced with:
 * - R: Center alpha value
 * - G: Maximum neighbor alpha value
 * - B: Alpha difference
 * @param {boolean} enabled
 */
GaussianSplatPrimitive.prototype.setOutlineDebugMode = function (enabled) {
  this._outlineDebugMode = enabled === true;
  if (this._outlineDebugMode) {
    console.log("[轮廓调试] 调试可视化模式已启用");
    console.log("[轮廓调试] 轮廓颜色将被替换为调试信息:");
    console.log("  - R (红色): 中心 alpha 值");
    console.log("  - G (绿色): 最大邻域 alpha 值");
    console.log("  - B (蓝色): Alpha 差值");
  } else {
    console.log("[轮廓调试] 调试可视化模式已禁用");
  }
  this._outlineParamsChanged = true;
  this._dirty = true;
};

/**
 * Sets the selection state of splats by their PLY indices.
 * This method converts PLY indices to aggregate indices using the internal mapping.
 * @param {Array<number>|Set<number>} plyIndices The PLY indices of splats to select.
 * @param {boolean} [selected=true] Whether to select (true) or deselect (false) the splats.
 */
GaussianSplatPrimitive.prototype.setSplatSelectionByPlyIndex = function (
  plyIndices,
  selected,
) {
  selected = selected !== false; // default to true
  const plyIndicesArray = Array.isArray(plyIndices)
    ? plyIndices
    : Array.from(plyIndices);

  if (!this._plyIndexSelectionLogShown) {
    console.log("[选中高亮-PLY] setSplatSelectionByPlyIndex 调用:");
    console.log("  - 总 splat 数:", this._numSplats);
    console.log("  - 输入 PLY 索引数量:", plyIndicesArray.length);
    console.log(
      "  - PLY 索引映射存在:",
      this._plyIndexToAggregateIndex.size > 0,
    );
    this._plyIndexSelectionLogShown = true;
  }

  let validCount = 0;
  let notFoundCount = 0;
  const aggregateIndices = [];

  for (let i = 0; i < plyIndicesArray.length; i++) {
    const plyIndex = plyIndicesArray[i];
    const aggregateIndex = this._plyIndexToAggregateIndex.get(plyIndex);
    if (
      defined(aggregateIndex) &&
      aggregateIndex >= 0 &&
      aggregateIndex < this._numSplats
    ) {
      aggregateIndices.push(aggregateIndex);
      validCount++;
    } else {
      notFoundCount++;
      if (notFoundCount <= 5) {
        console.warn(
          `[选中高亮-PLY] ⚠️ PLY 索引 ${plyIndex} 未找到对应的聚合索引`,
        );
      }
    }
  }

  if (validCount > 0) {
    // 使用聚合索引调用原有的选中方法
    this.setSplatSelection(aggregateIndices, selected);
    console.log(`[选中高亮-PLY] 成功转换 ${validCount} 个 PLY 索引为聚合索引`);
  }

  if (notFoundCount > 0) {
    console.warn(`[选中高亮-PLY] ⚠️ ${notFoundCount} 个 PLY 索引未找到映射`);
    console.warn(
      `[选中高亮-PLY] 提示: 确保 tileset 已加载完成，且包含 PLY 索引映射`,
    );
  }
};

/**
 * Sets the lock state of splats by their PLY indices.
 * This method converts PLY indices to aggregate indices using the internal mapping.
 * @param {Array<number>|Set<number>} plyIndices The PLY indices of splats to lock.
 * @param {boolean} [locked=true] Whether to lock (true) or unlock (false) the splats.
 */
GaussianSplatPrimitive.prototype.setSplatLockByPlyIndex = function (
  plyIndices,
  locked,
) {
  locked = locked !== false; // default to true
  const plyIndicesArray = Array.isArray(plyIndices)
    ? plyIndices
    : Array.from(plyIndices);

  if (!this._plyIndexLockLogShown) {
    console.log("[锁定高亮-PLY] setSplatLockByPlyIndex 调用:");
    console.log("  - 输入 PLY 索引数量:", plyIndicesArray.length);
    this._plyIndexLockLogShown = true;
  }

  let validCount = 0;
  let notFoundCount = 0;
  const aggregateIndices = [];

  for (let i = 0; i < plyIndicesArray.length; i++) {
    const plyIndex = plyIndicesArray[i];
    const aggregateIndex = this._plyIndexToAggregateIndex.get(plyIndex);
    if (
      defined(aggregateIndex) &&
      aggregateIndex >= 0 &&
      aggregateIndex < this._numSplats
    ) {
      aggregateIndices.push(aggregateIndex);
      validCount++;
    } else {
      notFoundCount++;
    }
  }

  if (validCount > 0) {
    // 使用聚合索引调用原有的锁定方法
    this.setSplatLock(aggregateIndices, locked);
    console.log(`[锁定高亮-PLY] 成功转换 ${validCount} 个 PLY 索引为聚合索引`);
  }

  if (notFoundCount > 0) {
    console.warn(`[锁定高亮-PLY] ⚠️ ${notFoundCount} 个 PLY 索引未找到映射`);
  }
};

/**
 * Since we aren't visible at the scene level, we need to wrap the tileset update
 * so we not only get called but ensure we update immediately after the tileset.
 * @param {FrameState} frameState
 * @private
 *
 */
GaussianSplatPrimitive.prototype._wrappedUpdate = function (frameState) {
  this._baseTilesetUpdate.call(this._tileset, frameState);
  this.update(frameState);
};

/**
 * Destroys the primitive and releases its resources in a deterministic manner.
 * @private
 */
GaussianSplatPrimitive.prototype.destroy = function () {
  if (defined(this._pickId)) {
    this._pickId.destroy();
    this._pickId = undefined;
  }
  this._positions = undefined;
  this._rotations = undefined;
  this._scales = undefined;
  this._colors = undefined;
  this._indexes = undefined;
  if (defined(this.gaussianSplatTexture)) {
    this.gaussianSplatTexture.destroy();
    this.gaussianSplatTexture = undefined;
  }

  if (defined(this._splatStateTexture)) {
    this._splatStateTexture.destroy();
    this._splatStateTexture = undefined;
  }

  const drawCommand = this._drawCommand;
  if (defined(drawCommand)) {
    drawCommand.shaderProgram =
      drawCommand.shaderProgram && drawCommand.shaderProgram.destroy();
  }

  if (defined(this._vertexArray)) {
    this._vertexArray.destroy();
    this._vertexArray = undefined;
  }

  destroyOutlineResources(this);

  this._tileset.update = this._baseTilesetUpdate.bind(this._tileset);

  return destroyObject(this);
};

/**
 * Returns true if this object was destroyed; otherwise, false.
 * <br /><br />
 * If this object was destroyed, it should not be used; calling any function other than
 * <code>isDestroyed</code> will result in a {@link DeveloperError} exception.
 * @returns {boolean} Returns true if the primitive has been destroyed, otherwise false.
 * @private
 */
GaussianSplatPrimitive.prototype.isDestroyed = function () {
  return this._isDestroyed;
};

/**
 * Get the PLY start index for a tile from its extras or compute from tileset.
 * @param {Cesium3DTile} tile The tile to get PLY index for.
 * @returns {number} The PLY start index for this tile.
 * @private
 */
GaussianSplatPrimitive.prototype._getTilePlyStartIndex = function (tile) {
  // 尝试从tile的extras属性中获取PLY索引范围（从tileset.json中读取）
  if (defined(tile.extras) && defined(tile.extras.plyIndexRange)) {
    const start = tile.extras.plyIndexRange.start;
    if (defined(start) && typeof start === "number") {
      return start;
    }
  }

  // 如果tile有_header.extras（备用方法）
  if (defined(tile._header) && defined(tile._header.extras)) {
    const extras = tile._header.extras;
    if (defined(extras.plyIndexRange) && defined(extras.plyIndexRange.start)) {
      return extras.plyIndexRange.start;
    }
  }

  // 作为fallback，尝试从已加载的tile中计算累积索引
  // 这需要遍历tileset的所有tile来计算
  // 暂时返回0，实际使用时应该根据tileset结构计算
  return 0;
};

/**
 * Event callback for when a tile is loaded.
 * This method is called when a tile is loaded and the primitive needs to be updated.
 * It sets the dirty flag to true, indicating that the primitive needs to be rebuilt.
 * @param {Cesium3DTile} tile
 * @private
 */
GaussianSplatPrimitive.prototype.onTileLoad = function (tile) {
  this._dirty = true;
};

/**
 * Callback for visible tiles.
 * @param {Cesium3DTile} tile
 * @private
 */
GaussianSplatPrimitive.prototype.onTileVisible = function (tile) {};

/**
 * Transforms the tile's splat primitive attributes into world space.
 * <br /><br />
 * This method applies the computed transform of the tile and the tileset's bounding sphere
 * to the splat primitive's position, rotation, and scale attributes.
 * It modifies the attributes in place, transforming them from local space to world space.
 *
 * @param {Cesium3DTile} tile
 * @private
 */
GaussianSplatPrimitive.transformTile = function (tile) {
  const computedTransform = tile.computedTransform;
  const gltfPrimitive = tile.content.gltfPrimitive;
  const gaussianSplatPrimitive = tile.tileset.gaussianSplatPrimitive;

  if (gaussianSplatPrimitive._rootTransform === undefined) {
    gaussianSplatPrimitive._rootTransform = Transforms.eastNorthUpToFixedFrame(
      tile.tileset.boundingSphere.center,
    );
  }
  const rootTransform = gaussianSplatPrimitive._rootTransform;

  const computedModelMatrix = Matrix4.multiplyTransformation(
    computedTransform,
    gaussianSplatPrimitive._axisCorrectionMatrix,
    scratchMatrix4A,
  );

  Matrix4.multiplyTransformation(
    computedModelMatrix,
    tile.content.worldTransform,
    computedModelMatrix,
  );

  const toGlobal = Matrix4.multiply(
    tile.tileset.modelMatrix,
    rootTransform,
    scratchMatrix4B,
  );
  const toLocal = Matrix4.inverse(toGlobal, scratchMatrix4C);
  const transform = Matrix4.multiplyTransformation(
    toLocal,
    computedModelMatrix,
    scratchMatrix4A,
  );
  const positions = tile.content.positions;
  const rotations = tile.content.rotations;
  const scales = tile.content.scales;
  const attributePositions = ModelUtility.getAttributeBySemantic(
    gltfPrimitive,
    VertexAttributeSemantic.POSITION,
  ).typedArray;

  const attributeRotations = ModelUtility.getAttributeBySemantic(
    gltfPrimitive,
    VertexAttributeSemantic.ROTATION,
  ).typedArray;

  const attributeScales = ModelUtility.getAttributeBySemantic(
    gltfPrimitive,
    VertexAttributeSemantic.SCALE,
  ).typedArray;

  const position = new Cartesian3();
  const rotation = new Quaternion();
  const scale = new Cartesian3();
  for (let i = 0; i < attributePositions.length / 3; ++i) {
    position.x = attributePositions[i * 3];
    position.y = attributePositions[i * 3 + 1];
    position.z = attributePositions[i * 3 + 2];

    rotation.x = attributeRotations[i * 4];
    rotation.y = attributeRotations[i * 4 + 1];
    rotation.z = attributeRotations[i * 4 + 2];
    rotation.w = attributeRotations[i * 4 + 3];

    scale.x = attributeScales[i * 3];
    scale.y = attributeScales[i * 3 + 1];
    scale.z = attributeScales[i * 3 + 2];

    Matrix4.fromTranslationQuaternionRotationScale(
      position,
      rotation,
      scale,
      scratchMatrix4C,
    );

    Matrix4.multiplyTransformation(transform, scratchMatrix4C, scratchMatrix4C);

    Matrix4.getTranslation(scratchMatrix4C, position);
    Matrix4.getRotation(scratchMatrix4C, rotation);
    Matrix4.getScale(scratchMatrix4C, scale);

    positions[i * 3] = position.x;
    positions[i * 3 + 1] = position.y;
    positions[i * 3 + 2] = position.z;

    rotations[i * 4] = rotation.x;
    rotations[i * 4 + 1] = rotation.y;
    rotations[i * 4 + 2] = rotation.z;
    rotations[i * 4 + 3] = rotation.w;

    scales[i * 3] = scale.x;
    scales[i * 3 + 1] = scale.y;
    scales[i * 3 + 2] = scale.z;
  }
};

/**
 * Updates the splat state texture based on current selection and lock states.
 * @param {GaussianSplatPrimitive} primitive
 * @param {FrameState} frameState
 * @private
 */
GaussianSplatPrimitive.updateSplatStateTexture = function (
  primitive,
  frameState,
) {
  if (primitive._numSplats === 0) {
    console.warn("[状态纹理] numSplats 为 0，跳过更新");
    return;
  }

  if (!primitive._stateTextureUpdateCount) {
    primitive._stateTextureUpdateCount = 0;
  }
  primitive._stateTextureUpdateCount++;

  if (primitive._stateTextureUpdateCount === 1) {
    console.log("[状态纹理] updateSplatStateTexture 首次调用:");
    console.log("  - numSplats:", primitive._numSplats);
    console.log("  - 选中数量:", primitive._selectedSplatIndices.size);
    console.log("  - 锁定数量:", primitive._lockedSplatIndices.size);
  }

  // Initialize state array if needed
  if (
    !defined(primitive._splatStates) ||
    primitive._splatStates.length < primitive._numSplats
  ) {
    primitive._splatStates = new Uint8Array(primitive._numSplats);
    // Initialize all states to 0
    primitive._splatStates.fill(0);
  }

  // Calculate texture dimensions (similar to how splat texture is organized)
  // Use the same width as the splat texture for consistency, or default to 1024
  // The state texture should match the splat texture layout for coordinate calculation
  const textureWidth =
    primitive._lastTextureWidth > 0 ? primitive._lastTextureWidth : 1024;
  const textureHeight = Math.ceil(primitive._numSplats / textureWidth);
  const textureSize = textureWidth * textureHeight;

  // Ensure state array is large enough for the texture
  // Texture size may be larger than numSplats due to rounding up
  if (
    !defined(primitive._splatStates) ||
    primitive._splatStates.length < textureSize
  ) {
    primitive._splatStates = new Uint8Array(textureSize);
    // Initialize all states to 0
    primitive._splatStates.fill(0);
  }

  // Update states based on selected, locked indices, and color groups
  // State encoding: bits 0-1 = flags (selected, locked), bits 2-7 = color group ID
  // Only update the first numSplats elements, rest remain 0
  let selectedCount = 0;
  let lockedCount = 0;
  for (let i = 0; i < primitive._numSplats; i++) {
    let state = 0;

    // Get color group ID (default to 0 if not assigned)
    const colorGroupId = primitive._splatColorGroups.get(i) || 0;
    // Clamp color group ID to valid range (0-63)
    const clampedGroupId = Math.min(
      Math.max(0, colorGroupId),
      primitive._maxColorGroups - 1,
    );
    // Store color group ID in bits 2-7
    state |= clampedGroupId << 2;

    if (primitive._selectedSplatIndices.has(i)) {
      state |= 1; // bit 0 = selected
      selectedCount++;
    }
    if (primitive._lockedSplatIndices.has(i)) {
      state |= 2; // bit 1 = locked
      lockedCount++;
    }
    primitive._splatStates[i] = state;
  }

  if (selectedCount > 0 || lockedCount > 0) {
    console.log(
      `[状态纹理] 状态更新完成: 选中=${selectedCount}, 锁定=${lockedCount}`,
    );

    // 找到第一个被选中的索引位置
    let firstSelectedIndex = -1;
    for (let i = 0; i < primitive._numSplats; i++) {
      if (primitive._splatStates[i] & 1) {
        firstSelectedIndex = i;
        break;
      }
    }

    if (firstSelectedIndex >= 0) {
      // 打印第一个被选中索引附近的状态值
      const startIdx = Math.max(0, firstSelectedIndex - 2);
      const endIdx = Math.min(primitive._numSplats, firstSelectedIndex + 8);
      const sampleStates = Array.from(
        primitive._splatStates.slice(startIdx, endIdx),
      );
      console.log(
        `[状态纹理] 第一个选中索引附近的状态值 (索引 ${startIdx}-${endIdx - 1}):`,
        sampleStates,
      );
    } else {
      // 如果没有找到，打印前几个状态值
      const sampleSize = Math.min(10, primitive._numSplats);
      const sampleStates = Array.from(
        primitive._splatStates.slice(0, sampleSize),
      );
      console.log(`[状态纹理] 前${sampleSize}个状态值:`, sampleStates);
    }
  }

  // Ensure remaining elements are 0 (in case texture is larger than numSplats)
  for (let i = primitive._numSplats; i < textureSize; i++) {
    primitive._splatStates[i] = 0;
  }

  // Create or update the state texture
  if (!defined(primitive._splatStateTexture)) {
    if (primitive._stateTextureUpdateCount === 1) {
      console.log(
        `[状态纹理] 创建新纹理: ${textureWidth}x${textureHeight}, 数据大小: ${primitive._splatStates.length}`,
      );
    }
    primitive._splatStateTexture = createSplatStateTexture(
      frameState.context,
      primitive._splatStates,
      textureWidth,
      textureHeight,
    );
    if (primitive._stateTextureUpdateCount === 1) {
      console.log("[状态纹理] ✓ 状态纹理创建成功");
    }
  } else if (
    primitive._lastTextureWidth !== textureWidth ||
    primitive._lastTextureHeight !== textureHeight
  ) {
    // Texture size changed, recreate it
    console.log(
      `[状态纹理] 纹理尺寸变化，重新创建: ${textureWidth}x${textureHeight}`,
    );
    const oldTex = primitive._splatStateTexture;
    primitive._splatStateTexture = createSplatStateTexture(
      frameState.context,
      primitive._splatStates,
      textureWidth,
      textureHeight,
    );
    oldTex.destroy();
    console.log("[状态纹理] ✓ 状态纹理重新创建成功");
  } else {
    // Update existing texture
    if (primitive._stateTextureUpdateCount <= 3) {
      console.log(
        `[状态纹理] 更新现有纹理 (第${primitive._stateTextureUpdateCount}次)`,
      );
    }
    primitive._splatStateTexture.copyFrom({
      source: {
        width: textureWidth,
        height: textureHeight,
        arrayBufferView: primitive._splatStates,
      },
    });
    if (primitive._stateTextureUpdateCount <= 3) {
      console.log("[状态纹理] ✓ 状态纹理更新成功");
    }
  }

  primitive._needsStateTextureUpdate = false;
};

/**
 * Generates the Gaussian splat texture for the primitive.
 * This method creates a texture from the splat attributes (positions, scales, rotations, colors)
 * and updates the primitive's state accordingly.
 *
 * @see {@link GaussianSplatTextureGenerator}
 *
 * @param {GaussianSplatPrimitive} primitive
 * @param {FrameState} frameState
 * @private
 */
GaussianSplatPrimitive.generateSplatTexture = function (primitive, frameState) {
  primitive._gaussianSplatTexturePending = true;

  // 复制颜色数组
  const colors = new Uint8Array(primitive._colors);

  // ========================================
  // [方案1] 应用颜色修改（新方法）
  // ========================================
  if (primitive._colorModifications.size > 0) {
    console.log(`\n[方案1-ColorMod] 开始应用颜色修改...`);
    console.log(
      `[方案1-ColorMod]   - 待修改数量: ${primitive._colorModifications.size}`,
    );
    console.log(`[方案1-ColorMod]   - 总 splat 数: ${primitive._numSplats}`);
    console.log(
      `[方案1-ColorMod]   - PLY 索引数组存在: ${defined(primitive._plyIndicesAggregate)}`,
    );

    const isRgba = colors.length === primitive._numSplats * 4;
    const componentsPerColor = isRgba ? 4 : 3;
    let appliedCount = 0;
    let notFoundCount = 0;

    if (defined(primitive._plyIndicesAggregate)) {
      // [方案1] 新方法：遍历所有 splat，查找需要修改颜色的
      console.log(`[方案1-ColorMod] 使用新方法（基于聚合 PLY 索引数组）`);

      for (
        let aggregateIndex = 0;
        aggregateIndex < primitive._numSplats;
        aggregateIndex++
      ) {
        const plyIndex = primitive._plyIndicesAggregate[aggregateIndex];

        // 检查是否有颜色修改
        if (primitive._colorModifications.has(plyIndex)) {
          const modifiedColor = primitive._colorModifications.get(plyIndex);
          const colorIndex = aggregateIndex * componentsPerColor;

          colors[colorIndex] = modifiedColor[0]; // R
          colors[colorIndex + 1] = modifiedColor[1]; // G
          colors[colorIndex + 2] = modifiedColor[2]; // B
          if (isRgba && modifiedColor.length > 3) {
            colors[colorIndex + 3] = modifiedColor[3]; // A
          }

          appliedCount++;

          // 调试：输出前几个修改
          if (appliedCount <= 5) {
            console.log(
              `[方案1-ColorMod]   修改 ${appliedCount}: aggregateIndex=${aggregateIndex}, plyIndex=${plyIndex}, color=[${modifiedColor.join(",")}]`,
            );
          }
        }
      }

      console.log(
        `[方案1-ColorMod] ✓ 颜色修改应用完成: ${appliedCount} 个成功`,
      );
    } else {
      // 回退到旧方法（使用映射）
      console.warn(
        `[方案1-ColorMod] ⚠️ PLY 索引数组不存在，使用旧方法（映射）`,
      );

      for (const [plyIndex, modifiedColor] of primitive._colorModifications) {
        const aggregateIndex =
          primitive._plyIndexToAggregateIndex.get(plyIndex);
        if (defined(aggregateIndex) && aggregateIndex < primitive._numSplats) {
          const colorIndex = aggregateIndex * componentsPerColor;
          colors[colorIndex] = modifiedColor[0]; // R
          colors[colorIndex + 1] = modifiedColor[1]; // G
          colors[colorIndex + 2] = modifiedColor[2]; // B
          if (isRgba && modifiedColor.length > 3) {
            colors[colorIndex + 3] = modifiedColor[3]; // A
          }
          appliedCount++;
        } else {
          notFoundCount++;
        }
      }

      console.log(
        `[方案1-ColorMod] 颜色修改应用: ${appliedCount} 个成功, ${notFoundCount} 个失败`,
      );
    }
  }

  // ========================================
  // 自定义滤镜（颜色处理管线）
  // ========================================
  if (
    defined(primitive._customColorFilters) &&
    primitive._customColorFilters.length > 0
  ) {
    const componentsPerColor =
      colors.length === primitive._numSplats * 4 ? 4 : 3;
    const isRgba = componentsPerColor === 4;
    applyCustomColorFilters(primitive, colors, componentsPerColor, isRgba);
  }

  // ========================================
  // [方案1] 传递 PLY 索引给纹理生成器
  // ========================================
  console.log(`\n[方案1] 准备生成纹理...`);
  console.log(`[方案1]   - splat 数量: ${primitive._numSplats}`);
  console.log(
    `[方案1]   - PLY 索引数组存在: ${defined(primitive._plyIndicesAggregate)}`,
  );
  if (defined(primitive._plyIndicesAggregate)) {
    console.log(
      `[方案1]   - PLY 索引数组长度: ${primitive._plyIndicesAggregate.length}`,
    );
    console.log(
      `[方案1]   - PLY 索引前5个: [${Array.from(primitive._plyIndicesAggregate.slice(0, 5)).join(", ")}]`,
    );
  }

  // [方案1] 创建 PLY 索引数组的副本，避免 ArrayBuffer 被 transfer 后分离
  // 注意：我们需要保留主线程的副本，因为后续可能需要重新生成纹理
  const plyIndicesCopy = defined(primitive._plyIndicesAggregate)
    ? new Uint32Array(primitive._plyIndicesAggregate)
    : undefined;

  if (defined(plyIndicesCopy)) {
    console.log(
      `[方案1]   ✓ PLY 索引副本已创建，长度: ${plyIndicesCopy.length}`,
    );
  }

  const promise = GaussianSplatTextureGenerator.generateFromAttributes({
    attributes: {
      positions: new Float32Array(primitive._positions),
      scales: new Float32Array(primitive._scales),
      rotations: new Float32Array(primitive._rotations),
      colors: colors, // 使用修改后的颜色
      plyIndices: plyIndicesCopy, // [方案1] 使用副本，避免原数组的 buffer 被分离
    },
    count: primitive._numSplats,
  });
  if (!defined(promise)) {
    primitive._gaussianSplatTexturePending = false;
    return;
  }
  promise
    .then((splatTextureData) => {
      if (!primitive._gaussianSplatTexture) {
        // First frame, so create the texture.
        primitive.gaussianSplatTexture = createGaussianSplatTexture(
          frameState.context,
          splatTextureData,
        );
      } else if (
        primitive._lastTextureHeight !== splatTextureData.height ||
        primitive._lastTextureWidth !== splatTextureData.width
      ) {
        const oldTex = primitive.gaussianSplatTexture;
        primitive._gaussianSplatTexture = createGaussianSplatTexture(
          frameState.context,
          splatTextureData,
        );
        oldTex.destroy();
      } else {
        primitive.gaussianSplatTexture.copyFrom({
          source: {
            width: splatTextureData.width,
            height: splatTextureData.height,
            arrayBufferView: splatTextureData.data,
          },
        });
      }
      primitive._vertexArray = undefined;
      primitive._lastTextureHeight = splatTextureData.height;
      primitive._lastTextureWidth = splatTextureData.width;

      primitive._hasGaussianSplatTexture = true;
      primitive._needsGaussianSplatTexture = false;
      primitive._gaussianSplatTexturePending = false;

      if (
        !defined(primitive._indexes) ||
        primitive._indexes.length < primitive._numSplats
      ) {
        primitive._indexes = new Uint32Array(primitive._numSplats);
      }
      for (let i = 0; i < primitive._numSplats; ++i) {
        primitive._indexes[i] = i;
      }
    })
    .catch((error) => {
      console.error("Error generating Gaussian splat texture:", error);
      primitive._gaussianSplatTexturePending = false;
    });
};

/**
 * Builds the draw command for the Gaussian splat primitive.
 * This method sets up the shader program, render state, and vertex array for rendering the Gaussian splats.
 * It also configures the attributes and uniforms required for rendering.
 *
 * @param {GaussianSplatPrimitive} primitive
 * @param {FrameState} frameState
 *
 * @private
 */
GaussianSplatPrimitive.buildGSplatDrawCommand = function (
  primitive,
  frameState,
) {
  const tileset = primitive._tileset;
  const renderResources = new GaussianSplatRenderResources(primitive);
  const { shaderBuilder } = renderResources;
  const renderStateOptions = renderResources.renderStateOptions;
  renderStateOptions.cull.enabled = false;
  renderStateOptions.depthMask = false;
  renderStateOptions.depthTest.enabled = true;
  renderStateOptions.blending = BlendingState.PRE_MULTIPLIED_ALPHA_BLEND;
  renderResources.alphaOptions.pass = Pass.GAUSSIAN_SPLATS;

  shaderBuilder.addAttribute("vec2", "a_screenQuadPosition");
  shaderBuilder.addAttribute("float", "a_splatIndex");
  shaderBuilder.addVarying("vec4", "v_splatColor");
  shaderBuilder.addVarying("vec2", "v_vertPos");
  shaderBuilder.addUniform(
    "float",
    "u_splitDirection",
    ShaderDestination.VERTEX,
  );
  shaderBuilder.addVarying("float", "v_splitDirection");

  // Add splat state support
  if (defined(primitive._splatStateTexture)) {
    if (!primitive._shaderBuildCount) {
      primitive._shaderBuildCount = 0;
    }
    primitive._shaderBuildCount++;

    if (primitive._shaderBuildCount === 1) {
      console.log("[着色器构建] 添加状态纹理支持");
    }
    shaderBuilder.addDefine("HAS_SPLAT_STATE", "1", ShaderDestination.BOTH);
    shaderBuilder.addVarying("float", "v_splatState");
    shaderBuilder.addUniform(
      "sampler2D",
      "u_splatStateTexture",
      ShaderDestination.VERTEX,
    );
    shaderBuilder.addUniform(
      "float",
      "u_stateTextureWidth",
      ShaderDestination.VERTEX,
    );
    shaderBuilder.addUniform(
      "vec4",
      "u_selectedColor",
      ShaderDestination.FRAGMENT,
    );
    shaderBuilder.addUniform(
      "vec4",
      "u_lockedColor",
      ShaderDestination.FRAGMENT,
    );
    shaderBuilder.addUniform(
      "vec4",
      "u_outlineColor",
      ShaderDestination.FRAGMENT,
    );
    shaderBuilder.addUniform(
      "float",
      "u_outlineAlphaCutoff",
      ShaderDestination.FRAGMENT,
    );
    shaderBuilder.addUniform(
      "float",
      "u_outlineWidth",
      ShaderDestination.FRAGMENT,
    );
    shaderBuilder.addUniform(
      "vec2",
      "u_outlineTexelStep",
      ShaderDestination.FRAGMENT,
    );
    shaderBuilder.addUniform(
      "bool",
      "u_outlineDepthTest",
      ShaderDestination.FRAGMENT,
    );
    // Add color groups uniform array (max 64 groups)
    shaderBuilder.addUniform(
      "vec4",
      "u_colorGroupColors[64]",
      ShaderDestination.FRAGMENT,
    );
    shaderBuilder.addUniform(
      "int",
      "u_colorGroupCount",
      ShaderDestination.FRAGMENT,
    );
    if (primitive._shaderBuildCount === 1) {
      console.log(
        "[着色器构建] ✓ HAS_SPLAT_STATE 已定义，uniform 已添加（包含颜色组支持）",
      );
    }
  } else if (!primitive._shaderBuildCount) {
    console.warn("[着色器构建] ⚠️ 状态纹理不存在，未添加状态支持");
    primitive._shaderBuildCount = 0;
  }
  shaderBuilder.addUniform(
    "bool",
    "u_outlineMaskPass",
    ShaderDestination.FRAGMENT,
  );
  shaderBuilder.addUniform(
    "bool",
    "u_outlineRingPass",
    ShaderDestination.FRAGMENT,
  );
  shaderBuilder.addUniform(
    "highp usampler2D",
    "u_splatAttributeTexture",
    ShaderDestination.VERTEX,
  );

  shaderBuilder.addUniform(
    "float",
    "u_sphericalHarmonicsDegree",
    ShaderDestination.VERTEX,
  );

  shaderBuilder.addUniform("float", "u_splatScale", ShaderDestination.VERTEX);

  shaderBuilder.addUniform(
    "vec3",
    "u_cameraPositionWC",
    ShaderDestination.VERTEX,
  );

  shaderBuilder.addUniform(
    "mat3",
    "u_inverseModelRotation",
    ShaderDestination.VERTEX,
  );

  const uniformMap = renderResources.uniformMap;

  const textureCache = primitive.gaussianSplatTexture;
  uniformMap.u_splatAttributeTexture = function () {
    return textureCache;
  };

  if (primitive._sphericalHarmonicsDegree > 0) {
    shaderBuilder.addDefine(
      "HAS_SPHERICAL_HARMONICS",
      "1",
      ShaderDestination.VERTEX,
    );
    shaderBuilder.addUniform(
      "highp usampler2D",
      "u_sphericalHarmonicsTexture",
      ShaderDestination.VERTEX,
    );
    uniformMap.u_sphericalHarmonicsTexture = function () {
      return primitive.sphericalHarmonicsTexture;
    };
  }
  uniformMap.u_sphericalHarmonicsDegree = function () {
    return primitive._sphericalHarmonicsDegree;
  };

  uniformMap.u_cameraPositionWC = function () {
    return Cartesian3.clone(frameState.camera.positionWC);
  };

  uniformMap.u_inverseModelRotation = function () {
    const tileset = primitive._tileset;
    const modelMatrix = Matrix4.multiply(
      tileset.modelMatrix,
      primitive._rootTransform,
      scratchMatrix4A,
    );
    const inverseModelRotation = Matrix4.getRotation(
      Matrix4.inverse(modelMatrix, scratchMatrix4C),
      scratchMatrix4D,
    );
    return inverseModelRotation;
  };

  uniformMap.u_splitDirection = function () {
    return primitive.splitDirection;
  };

  // Add state texture and color uniforms
  if (defined(primitive._splatStateTexture)) {
    if (!primitive._uniformMapCount) {
      primitive._uniformMapCount = 0;
    }
    primitive._uniformMapCount++;

    uniformMap.u_splatStateTexture = function () {
      return primitive._splatStateTexture;
    };
    uniformMap.u_stateTextureWidth = function () {
      return primitive._lastTextureWidth || 1024.0;
    };
    uniformMap.u_selectedColor = function () {
      return primitive._selectedColor;
    };
    uniformMap.u_lockedColor = function () {
      return primitive._lockedColor;
    };

    // Map color groups array
    uniformMap.u_colorGroupColors = function () {
      // Create array with default colors for all groups
      const colorArray = new Array(primitive._maxColorGroups);
      // Group 0 uses default selection color
      colorArray[0] = primitive._selectedColor;
      // Fill remaining groups
      for (let i = 1; i < primitive._maxColorGroups; i++) {
        if (primitive._colorGroups[i]) {
          colorArray[i] = primitive._colorGroups[i];
        } else {
          // Default color for unset groups (use selection color as fallback)
          colorArray[i] = primitive._selectedColor;
        }
      }
      return colorArray;
    };

    uniformMap.u_colorGroupCount = function () {
      return primitive._maxColorGroups;
    };

    if (primitive._uniformMapCount === 1) {
      console.log("[Uniform映射] ✓ 状态 uniform 已设置:");
      console.log(
        "  - selectedColor:",
        primitive._selectedColor.toCssColorString(),
      );
      console.log(
        "  - lockedColor:",
        primitive._lockedColor.toCssColorString(),
      );
      console.log(
        "  - stateTextureWidth:",
        primitive._lastTextureWidth || 1024.0,
      );
      console.log("  - colorGroupCount:", primitive._maxColorGroups);
    }
  } else if (!primitive._uniformMapCount) {
    console.warn("[Uniform映射] ⚠️ 状态纹理不存在，未设置 uniform");
    primitive._uniformMapCount = 0;
  }

  uniformMap.u_outlineMaskPass = function () {
    return false;
  };
  uniformMap.u_outlineRingPass = function () {
    return false;
  };
  uniformMap.u_outlineAlphaCutoff = function () {
    return primitive._outlineAlphaCutoff;
  };
  uniformMap.u_outlineColor = function () {
    return primitive._outlineColor;
  };
  uniformMap.u_outlineWidth = function () {
    return primitive._outlineWidth;
  };
  uniformMap.u_outlineTexelStep = function () {
    return primitive._outlineTexelStep;
  };
  uniformMap.u_outlineDepthTest = function () {
    return primitive._outlineDepthTestEnabled;
  };

  renderResources.instanceCount = primitive._numSplats;
  renderResources.count = 4;
  renderResources.primitiveType = PrimitiveType.TRIANGLE_STRIP;

  // Create pickId for picking support (before building shader)
  // Use the first selected tile's content as the pick object
  let pickId = primitive._pickId;
  if (!defined(pickId) && tileset._selectedTiles.length > 0) {
    const firstTile = tileset._selectedTiles[0];
    if (defined(firstTile) && defined(firstTile.content)) {
      const pickObject = {
        content: firstTile.content,
        primitive: tileset,
      };
      pickId = frameState.context.createPickId(pickObject);
      primitive._pickId = pickId;
      //>>includeStart('debug', pragmas.debug);
      console.log("[拾取流程] GaussianSplatPrimitive 创建 pickId");
      console.log("  - pickId.color:", pickId.color);
      console.log(
        "  - pickObject.content:",
        pickObject.content?.constructor?.name,
      );
      console.log(
        "  - pickObject.primitive:",
        pickObject.primitive?.constructor?.name,
      );
      //>>includeEnd('debug');
    }
  }

  // Add czm_pickColor uniform to shader if pickId exists (must be before building shader)
  if (defined(pickId)) {
    shaderBuilder.addUniform(
      "vec4",
      "czm_pickColor",
      ShaderDestination.FRAGMENT,
    );
    //>>includeStart('debug', pragmas.debug);
    console.log("[拾取流程] 添加 czm_pickColor uniform 到着色器");
    //>>includeEnd('debug');
  }

  shaderBuilder.addVertexLines(GaussianSplatVS);
  shaderBuilder.addFragmentLines(GaussianSplatFS);

  const shaderProgram = shaderBuilder.buildShaderProgram(frameState.context);

  // 验证着色器是否包含状态处理
  if (defined(primitive._splatStateTexture)) {
    if (!primitive._shaderVerifyCount) {
      primitive._shaderVerifyCount = 0;
    }
    primitive._shaderVerifyCount++;

    if (primitive._shaderVerifyCount === 1) {
      console.log("[着色器验证] 检查着色器程序:");
      console.log("  - shaderProgram 存在:", defined(shaderProgram));

      // 检查 shaderBuilder 的 define 列表
      const vertexDefines = shaderBuilder._vertexShaderParts.defineLines || [];
      const fragmentDefines =
        shaderBuilder._fragmentShaderParts.defineLines || [];
      const hasStateDefine =
        vertexDefines.some((line) => line.includes("HAS_SPLAT_STATE")) ||
        fragmentDefines.some((line) => line.includes("HAS_SPLAT_STATE"));

      // 检查 uniform 列表
      const vertexUniforms =
        shaderBuilder._vertexShaderParts.uniformLines || [];
      const fragmentUniforms =
        shaderBuilder._fragmentShaderParts.uniformLines || [];
      const hasStateTexture = vertexUniforms.some((line) =>
        line.includes("u_splatStateTexture"),
      );
      const hasSelectedColor = fragmentUniforms.some((line) =>
        line.includes("u_selectedColor"),
      );
      const hasLockedColor = fragmentUniforms.some((line) =>
        line.includes("u_lockedColor"),
      );

      console.log("  - HAS_SPLAT_STATE 定义存在:", hasStateDefine);
      console.log("  - u_splatStateTexture uniform 存在:", hasStateTexture);
      console.log("  - u_selectedColor uniform 存在:", hasSelectedColor);
      console.log("  - u_lockedColor uniform 存在:", hasLockedColor);

      if (
        hasStateDefine &&
        hasStateTexture &&
        hasSelectedColor &&
        hasLockedColor
      ) {
        console.log("[着色器验证] ✓ 所有状态相关的定义和 uniform 都已添加");

        // 检查编译后的着色器源代码是否包含状态处理代码
        if (
          defined(shaderProgram) &&
          defined(shaderProgram._vertexShaderSource)
        ) {
          // 着色器源代码可能是字符串或数组，需要转换为字符串
          let vertexSource = shaderProgram._vertexShaderSource;
          let fragmentSource = shaderProgram._fragmentShaderSource;

          // 如果是数组，连接成字符串
          if (Array.isArray(vertexSource)) {
            vertexSource = vertexSource.join("\n");
          }
          if (Array.isArray(fragmentSource)) {
            fragmentSource = fragmentSource.join("\n");
          }

          // 确保是字符串类型
          vertexSource = String(vertexSource || "");
          fragmentSource = String(fragmentSource || "");

          const vertexHasStateCode =
            vertexSource.includes("u_splatStateTexture") &&
            vertexSource.includes("v_splatState");
          const fragmentHasStateCode =
            fragmentSource.includes("v_splatState") &&
            (fragmentSource.includes("u_selectedColor") ||
              fragmentSource.includes("u_lockedColor")) &&
            (fragmentSource.includes("mix") ||
              fragmentSource.includes("finalColor"));

          console.log("[着色器验证] 检查编译后的着色器源代码:");
          console.log("  - 顶点着色器包含状态代码:", vertexHasStateCode);
          console.log("  - 片段着色器包含状态代码:", fragmentHasStateCode);

          if (!vertexHasStateCode) {
            console.error(
              "[着色器验证] ❌ 顶点着色器源代码中缺少状态处理代码！",
            );
            console.error(
              "  请检查 PrimitiveGaussianSplatVS.js 是否包含 HAS_SPLAT_STATE 块",
            );
          }
          if (!fragmentHasStateCode) {
            console.error(
              "[着色器验证] ❌ 片段着色器源代码中缺少状态处理代码！",
            );
            console.error(
              "  请检查 PrimitiveGaussianSplatFS.js 是否包含 HAS_SPLAT_STATE 块",
            );
          }

          if (vertexHasStateCode && fragmentHasStateCode) {
            console.log("[着色器验证] ✓ 着色器源代码包含完整的状态处理代码");
          }
        }
      } else {
        console.error("[着色器验证] ❌ 缺少状态相关的定义或 uniform！");
        if (!hasStateDefine) {
          console.error("  - 缺少 HAS_SPLAT_STATE 定义");
        }
        if (!hasStateTexture) {
          console.error("  - 缺少 u_splatStateTexture uniform");
        }
        if (!hasSelectedColor) {
          console.error("  - 缺少 u_selectedColor uniform");
        }
        if (!hasLockedColor) {
          console.error("  - 缺少 u_lockedColor uniform");
        }
      }
    }
  }

  let renderState = clone(
    RenderState.fromCache(renderResources.renderStateOptions),
    true,
  );

  renderState.cull.face = ModelUtility.getCullFace(
    tileset.modelMatrix,
    PrimitiveType.TRIANGLE_STRIP,
  );

  renderState = RenderState.fromCache(renderState);
  const splatQuadAttrLocations = {
    screenQuadPosition: 0,
    splatIndex: 2,
  };

  const idxAttr = new ModelComponents.Attribute();
  idxAttr.name = "_SPLAT_INDEXES";
  idxAttr.typedArray = primitive._indexes;
  idxAttr.componentDatatype = ComponentDatatype.UNSIGNED_INT;
  idxAttr.type = AttributeType.SCALAR;
  idxAttr.normalized = false;
  idxAttr.count = renderResources.instanceCount;
  idxAttr.constant = 0;
  idxAttr.instanceDivisor = 1;

  if (
    !defined(primitive._vertexArray) ||
    primitive._indexes.length > primitive._vertexArrayLen
  ) {
    const geometry = new Geometry({
      attributes: {
        screenQuadPosition: new GeometryAttribute({
          componentDatatype: ComponentDatatype.FLOAT,
          componentsPerAttribute: 2,
          values: [-1, -1, 1, -1, 1, 1, -1, 1],
          name: "_SCREEN_QUAD_POS",
          variableName: "screenQuadPosition",
        }),
        splatIndex: { ...idxAttr, variableName: "splatIndex" },
      },
      primitiveType: PrimitiveType.TRIANGLE_STRIP,
    });

    primitive._vertexArray = VertexArray.fromGeometry({
      context: frameState.context,
      geometry: geometry,
      attributeLocations: splatQuadAttrLocations,
      bufferUsage: BufferUsage.DYNAMIC_DRAW,
      interleave: false,
    });
  } else {
    primitive._vertexArray
      .getAttribute(1)
      .vertexBuffer.copyFromArrayView(primitive._indexes);
  }

  primitive._vertexArrayLen = primitive._indexes.length;

  const modelMatrix = Matrix4.multiply(
    tileset.modelMatrix,
    primitive._rootTransform,
    scratchMatrix4B,
  );

  const vertexArrayCache = primitive._vertexArray;

  const command = new DrawCommand({
    boundingVolume: tileset.boundingSphere,
    modelMatrix: modelMatrix,
    uniformMap: uniformMap,
    renderState: renderState,
    vertexArray: vertexArrayCache,
    shaderProgram: shaderProgram,
    cull: renderStateOptions.cull.enabled,
    pass: Pass.GAUSSIAN_SPLATS,
    count: renderResources.count,
    owner: this,
    instanceCount: renderResources.instanceCount,
    primitiveType: PrimitiveType.TRIANGLE_STRIP,
    debugShowBoundingVolume: tileset.debugShowBoundingVolume,
    castShadows: false,
    receiveShadows: false,
    pickId: defined(pickId) ? "czm_pickColor" : undefined,
  });

  // Set pick color uniform if pickId exists
  // Cesium will automatically create a derived pick command when pickId is set
  // We need to provide the czm_pickColor uniform
  if (defined(pickId)) {
    uniformMap.czm_pickColor = function () {
      return pickId.color;
    };
    //>>includeStart('debug', pragmas.debug);
    console.log("[拾取流程] 设置 pickId 到 drawCommand");
    console.log("  - pickId.color:", pickId.color);
    console.log("  - uniformMap.czm_pickColor 已设置");
    //>>includeEnd('debug');
  }

  primitive._drawCommand = command;
  primitive._outlineMaskCommand = undefined;
};

GaussianSplatPrimitive.buildOutlineMaskCommand = function (
  primitive,
  frameState,
) {
  if (!defined(primitive._drawCommand)) {
    return;
  }

  const baseCommand = primitive._drawCommand;
  const maskUniformMap = createOutlineMaskUniformMap(baseCommand.uniformMap);

  const maskRenderState = RenderState.fromCache({
    depthTest: {
      enabled: true,
    },
    depthMask: true,
    cull: {
      enabled: false,
    },
  });

  primitive._outlineMaskCommand = new DrawCommand({
    boundingVolume: baseCommand.boundingVolume,
    modelMatrix: baseCommand.modelMatrix,
    uniformMap: maskUniformMap,
    renderState: maskRenderState,
    shaderProgram: baseCommand.shaderProgram,
    vertexArray: primitive._vertexArray,
    cull: false,
    pass: baseCommand.pass,
    count: baseCommand.count,
    instanceCount: baseCommand.instanceCount,
    primitiveType: baseCommand.primitiveType,
    owner: primitive,
  });
};

GaussianSplatPrimitive.buildOutlineRingCommand = function (
  primitive,
  frameState,
) {
  if (!defined(primitive._drawCommand)) {
    return;
  }

  primitive._ensureBoundaryResources(frameState);
  if (
    !defined(primitive._boundaryVertexArray) ||
    primitive._boundaryInstanceCount === 0
  ) {
    primitive._outlineRingCommand = undefined;
    return;
  }

  const baseCommand = primitive._drawCommand;
  const ringUniformMap = createOutlineRingUniformMap(baseCommand.uniformMap);

  const ringRenderState = RenderState.fromCache({
    depthTest: {
      enabled: true,
    },
    depthMask: false,
    cull: {
      enabled: false,
    },
    blending: BlendingState.ALPHA_BLEND,
  });

  primitive._outlineRingCommand = new DrawCommand({
    boundingVolume: baseCommand.boundingVolume,
    modelMatrix: baseCommand.modelMatrix,
    uniformMap: ringUniformMap,
    renderState: ringRenderState,
    shaderProgram: baseCommand.shaderProgram,
    vertexArray: primitive._boundaryVertexArray,
    cull: false,
    pass: Pass.TRANSLUCENT,
    count: baseCommand.count,
    instanceCount: primitive._boundaryInstanceCount,
    primitiveType: baseCommand.primitiveType,
    owner: primitive,
  });
};

GaussianSplatPrimitive.prototype._ensureOutlineCompositeCommand = function (
  frameState,
) {
  // If outline parameters changed, recreate the composite command to ensure new parameters are used
  if (this._outlineParamsChanged && defined(this._outlineCompositeCommand)) {
    this._outlineCompositeCommand = undefined;
    this._outlineParamsChanged = false;
  }

  if (defined(this._outlineCompositeCommand)) {
    return;
  }

  const that = this;
  const fs = new ShaderSource({
    sources: [OutlineCompositeFS],
  });

  const outlineRenderState = RenderState.fromCache({
    depthMask: false,
    depthTest: {
      enabled: false,
    },
    blending: BlendingState.ALPHA_BLEND,
  });

  this._outlineCompositeCommand = frameState.context.createViewportQuadCommand(
    fs,
    {
      owner: this,
      pass: Pass.TRANSLUCENT,
      renderState: outlineRenderState,
      uniformMap: {
        u_outlineMaskTexture: function () {
          return that._outlineMaskTexture;
        },
        u_outlineColor: function () {
          return that._outlineColor;
        },
        u_outlineDepthTexture: function () {
          return that._outlineDepthTexture;
        },
        u_outlineDepthTest: function () {
          return (
            that._outlineDepthTestEnabled && defined(that._outlineDepthTexture)
          );
        },
        u_outlineHasDepth: function () {
          return defined(that._outlineDepthTexture);
        },
        u_outlineAlphaCutoff: function () {
          return that._outlineAlphaCutoff;
        },
        u_outlineMinAlphaDiff: function () {
          return that._outlineMinAlphaDiff;
        },
        u_outlineWidth: function () {
          return that._outlineWidth;
        },
        u_outlineTexelStep: function () {
          return that._outlineTexelStep;
        },
        u_outlineKernelRadius: function () {
          return that._outlineKernelRadius;
        },
        u_outlineDebugMode: function () {
          return that._outlineDebugMode || false;
        },
      },
    },
  );
  this._outlineCompositeCommand.boundingVolume = this._tileset.boundingSphere;
};

GaussianSplatPrimitive.prototype._queueOutlineCommands = function (frameState) {
  if (!this._outlineEnabled) {
    return;
  }

  if (!defined(this._drawCommand)) {
    return;
  }

  const baseCommand = this._drawCommand;
  if (this._outlineMode === "rings") {
    if (
      this._boundaryDirty ||
      !defined(this._outlineRingCommand) ||
      !defined(this._boundaryVertexArray)
    ) {
      GaussianSplatPrimitive.buildOutlineRingCommand(this, frameState);
    }

    if (!defined(this._outlineRingCommand)) {
      return;
    }

    this._outlineRingCommand.boundingVolume = baseCommand.boundingVolume;
    this._outlineRingCommand.modelMatrix = baseCommand.modelMatrix;
    frameState.commandList.push(this._outlineRingCommand);
    return;
  }

  updateOutlineFramebuffer(this, frameState);

  if (!defined(this._outlineMaskCommand)) {
    GaussianSplatPrimitive.buildOutlineMaskCommand(this, frameState);
  }

  if (!defined(this._outlineMaskCommand)) {
    return;
  }

  this._outlineMaskCommand.boundingVolume = baseCommand.boundingVolume;
  this._outlineMaskCommand.modelMatrix = baseCommand.modelMatrix;
  this._outlineMaskCommand.vertexArray = this._vertexArray;
  this._outlineMaskCommand.count = baseCommand.count;
  this._outlineMaskCommand.instanceCount = baseCommand.instanceCount;
  this._outlineMaskCommand.shaderProgram = baseCommand.shaderProgram;
  this._outlineMaskCommand.framebuffer = this._outlineFramebuffer.framebuffer;

  this._outlinePixelRatio = frameState.pixelRatio;
  this._outlineTexelStep.x = 1.0 / frameState.context.drawingBufferWidth;
  this._outlineTexelStep.y = 1.0 / frameState.context.drawingBufferHeight;

  this._outlineClearCommand.framebuffer = this._outlineFramebuffer.framebuffer;

  this._ensureOutlineCompositeCommand(frameState);
  if (defined(this._outlineCompositeCommand)) {
    this._outlineCompositeCommand.boundingVolume = this._tileset.boundingSphere;
  }

  frameState.commandList.push(this._outlineClearCommand);
  frameState.commandList.push(this._outlineMaskCommand);
  frameState.commandList.push(this._outlineCompositeCommand);
};

/**
 * Updates the Gaussian splat primitive for the current frame.
 * This method checks if the primitive needs to be updated based on the current frame state,
 * and if so, it processes the selected tiles, aggregates their attributes,
 * and generates the Gaussian splat texture if necessary.
 * It also handles the sorting of splat indexes and builds the draw command for rendering.
 *
 * @param {FrameState} frameState
 * @private
 */
GaussianSplatPrimitive.prototype.update = function (frameState) {
  const tileset = this._tileset;

  if (!tileset.show || tileset._selectedTiles.length === 0) {
    return;
  }

  // Ensure pickId is created before pick pass
  if (frameState.passes.pick === true) {
    //>>includeStart('debug', pragmas.debug);
    console.log("[拾取流程] GaussianSplatPrimitive.update() 拾取通道");
    console.log("  - _drawCommand 存在:", defined(this._drawCommand));
    console.log("  - _pickId 存在:", defined(this._pickId));
    //>>includeEnd('debug');

    // Create pickId if not exists
    if (!defined(this._pickId) && tileset._selectedTiles.length > 0) {
      const firstTile = tileset._selectedTiles[0];
      if (defined(firstTile) && defined(firstTile.content)) {
        const pickObject = {
          content: firstTile.content,
          primitive: tileset,
        };
        this._pickId = frameState.context.createPickId(pickObject);
        //>>includeStart('debug', pragmas.debug);
        console.log("[拾取流程] GaussianSplatPrimitive 在拾取通道创建 pickId");
        console.log("  - pickId.color:", this._pickId.color);
        //>>includeEnd('debug');

        // Update drawCommand pickId if it exists
        if (defined(this._drawCommand)) {
          this._drawCommand.pickId = "czm_pickColor";
          // Update uniform map
          const pickId = this._pickId;
          if (defined(this._drawCommand.uniformMap)) {
            this._drawCommand.uniformMap.czm_pickColor = function () {
              return pickId.color;
            };
            //>>includeStart('debug', pragmas.debug);
            console.log("[拾取流程] 更新 drawCommand 的 pickId 和 uniformMap");
            console.log("  - pickId.color:", pickId.color);
            //>>includeEnd('debug');
          }
        }
      }
    }

    if (this._drawCommand) {
      frameState.commandList.push(this._drawCommand);
    }
    return;
  }

  if (this._drawCommand) {
    frameState.commandList.push(this._drawCommand);
    if (this._outlineEnabled) {
      this._queueOutlineCommands(frameState);
    }
  }

  if (tileset._modelMatrixChanged) {
    this._dirty = true;
    return;
  }

  if (this.splitDirection !== tileset.splitDirection) {
    this.splitDirection = tileset.splitDirection;
  }

  if (this._sorterState === GaussianSplatSortingState.IDLE) {
    if (
      !this._dirty &&
      Matrix4.equals(frameState.camera.viewMatrix, this._prevViewMatrix)
    ) {
      // No need to update if the view matrix hasn't changed and the primitive isn't dirty.
      return;
    }

    if (
      tileset._selectedTiles.length !== 0 &&
      tileset._selectedTiles.length !== this.selectedTileLength
    ) {
      this._numSplats = 0;
      this._positions = undefined;
      this._rotations = undefined;
      this._scales = undefined;
      this._colors = undefined;
      this._indexes = undefined;
      this._shData = undefined;
      this._needsGaussianSplatTexture = true;
      this._gaussianSplatTexturePending = false;

      const tiles = tileset._selectedTiles;
      const totalElements = tiles.reduce(
        (total, tile) => total + tile.content.pointsLength,
        0,
      );

      // 建立PLY索引映射
      this._tilePlyIndexOffsets = [];
      this._plyIndexToAggregateIndex.clear();
      let aggregateIndexOffset = 0;

      for (const tile of tiles) {
        const tilePointsLength = tile.content.pointsLength;
        const plyIndexArray = tile.content.plyIndices;

        console.log(`处理 tile: ${tile._header.uri || "root"}`);
        console.log(`  - pointsLength: ${tilePointsLength}`);
        console.log(`  - plyIndices 存在: ${defined(plyIndexArray)}`);
        if (defined(plyIndexArray)) {
          console.log(`  - plyIndices 长度: ${plyIndexArray.length}`);
          console.log(
            `  - plyIndices 前10个: [${Array.from(plyIndexArray.slice(0, 10)).join(", ")}]`,
          );
        }

        // 优先使用 _PLY_INDEX attribute 建立精确映射
        if (
          defined(plyIndexArray) &&
          plyIndexArray.length === tilePointsLength
        ) {
          // 使用 _PLY_INDEX attribute 建立精确映射
          for (let i = 0; i < tilePointsLength; i++) {
            const plyIndex = plyIndexArray[i];
            const aggregateIndex = aggregateIndexOffset + i;
            this._plyIndexToAggregateIndex.set(plyIndex, aggregateIndex);
          }

          // 计算 PLY 索引范围（用于记录）
          let minIndex = Number.POSITIVE_INFINITY;
          let maxIndex = Number.NEGATIVE_INFINITY;
          for (let i = 0; i < tilePointsLength; i++) {
            const plyIndex = plyIndexArray[i];
            if (plyIndex < minIndex) {
              minIndex = plyIndex;
            }
            if (plyIndex > maxIndex) {
              maxIndex = plyIndex;
            }
          }

          this._tilePlyIndexOffsets.push({
            tile: tile,
            plyStartIndex: minIndex,
            plyEndIndex: maxIndex,
            aggregateStartIndex: aggregateIndexOffset,
            aggregateEndIndex: aggregateIndexOffset + tilePointsLength - 1,
          });

          console.log(
            `Tile ${tile._header.uri || "root"}: 使用 _PLY_INDEX attribute, 范围 [${minIndex}, ${maxIndex}], 点数 ${tilePointsLength}`,
          );
        } else {
          // 回退到旧的连续索引假设
          const plyStartIndex = this._getTilePlyStartIndex(tile);
          const plyEndIndex = plyStartIndex + tilePointsLength - 1;

          this._tilePlyIndexOffsets.push({
            tile: tile,
            plyStartIndex: plyStartIndex,
            plyEndIndex: plyEndIndex,
            aggregateStartIndex: aggregateIndexOffset,
            aggregateEndIndex: aggregateIndexOffset + tilePointsLength - 1,
          });

          // 建立PLY索引到聚合索引的映射
          // 假设tile内的点按顺序对应PLY索引（从plyStartIndex开始）
          for (let i = 0; i < tilePointsLength; i++) {
            const plyIndex = plyStartIndex + i;
            const aggregateIndex = aggregateIndexOffset + i;
            this._plyIndexToAggregateIndex.set(plyIndex, aggregateIndex);
          }

          console.warn(
            `Tile ${tile._header.uri || "root"}: 未找到 _PLY_INDEX attribute, 使用连续索引假设, 范围 [${plyStartIndex}, ${plyEndIndex}]`,
          );
        }

        aggregateIndexOffset += tilePointsLength;
      }

      // ========================================
      // [方案1] 聚合 PLY 索引数组
      // ========================================
      console.log(`\n[方案1] 开始聚合 PLY 索引数组...`);
      this._plyIndicesAggregate = new Uint32Array(totalElements);
      let plyIndexOffset = 0;

      for (const tile of tiles) {
        const content = tile.content;
        const tilePointsLength = content.pointsLength;

        if (
          defined(content.plyIndices) &&
          content.plyIndices.length === tilePointsLength
        ) {
          // 使用瓦片的 PLY 索引数组
          this._plyIndicesAggregate.set(content.plyIndices, plyIndexOffset);
          console.log(
            `[方案1] 瓦片 ${tile._header.uri || "root"}: 复制 ${tilePointsLength} 个 PLY 索引`,
          );
          console.log(`[方案1]   - 偏移: ${plyIndexOffset}`);
          console.log(
            `[方案1]   - 前5个: [${Array.from(content.plyIndices.slice(0, 5)).join(", ")}]`,
          );
        } else {
          // 回退：使用连续索引
          console.warn(
            `[方案1] 瓦片 ${tile._header.uri || "root"}: 未找到 PLY 索引，使用连续索引`,
          );
          for (let i = 0; i < tilePointsLength; i++) {
            this._plyIndicesAggregate[plyIndexOffset + i] = plyIndexOffset + i;
          }
        }

        plyIndexOffset += tilePointsLength;
      }

      console.log(`[方案1] ✓ PLY 索引聚合完成:`);
      console.log(`[方案1]   - 总数: ${this._plyIndicesAggregate.length}`);
      console.log(
        `[方案1]   - 前10个: [${Array.from(this._plyIndicesAggregate.slice(0, 10)).join(", ")}]`,
      );
      console.log(
        `[方案1]   - 最后10个: [${Array.from(this._plyIndicesAggregate.slice(-10)).join(", ")}]`,
      );

      const aggregateAttributeValues = (
        componentDatatype,
        getAttributeCallback,
        numberOfComponents,
      ) => {
        let aggregate;
        let offset = 0;
        for (const tile of tiles) {
          const content = tile.content;
          const attribute = getAttributeCallback(content);
          const componentsPerAttribute = defined(numberOfComponents)
            ? numberOfComponents
            : AttributeType.getNumberOfComponents(attribute.type);
          const buffer = defined(attribute.typedArray)
            ? attribute.typedArray
            : attribute;
          if (!defined(aggregate)) {
            aggregate = ComponentDatatype.createTypedArray(
              componentDatatype,
              totalElements * componentsPerAttribute,
            );
          }
          aggregate.set(buffer, offset);
          offset += buffer.length;
        }
        return aggregate;
      };

      const aggregateShData = () => {
        let offset = 0;
        for (const tile of tiles) {
          const shData = tile.content.packedSphericalHarmonicsData;
          if (tile.content.sphericalHarmonicsDegree > 0) {
            if (!defined(this._shData)) {
              let coefs;
              switch (tile.content.sphericalHarmonicsDegree) {
                case 1:
                  coefs = 9;
                  break;
                case 2:
                  coefs = 24;
                  break;
                case 3:
                  coefs = 45;
              }
              this._shData = new Uint32Array(totalElements * (coefs * (2 / 3)));
            }
            this._shData.set(shData, offset);
            offset += shData.length;
          }
        }
      };

      this._positions = aggregateAttributeValues(
        ComponentDatatype.FLOAT,
        (content) => content.positions,
        3,
      );

      this._scales = aggregateAttributeValues(
        ComponentDatatype.FLOAT,
        (content) => content.scales,
        3,
      );

      this._boundaryDirty = true;

      this._rotations = aggregateAttributeValues(
        ComponentDatatype.FLOAT,
        (content) => content.rotations,
        4,
      );

      this._colors = aggregateAttributeValues(
        ComponentDatatype.UNSIGNED_BYTE,
        (content) =>
          ModelUtility.getAttributeBySemantic(
            content.gltfPrimitive,
            VertexAttributeSemantic.COLOR,
          ),
      );

      aggregateShData();
      this._sphericalHarmonicsDegree =
        tiles[0].content.sphericalHarmonicsDegree;

      this._numSplats = totalElements;
      this.selectedTileLength = tileset._selectedTiles.length;

      // Initialize state array when splat count changes
      this._splatStates = new Uint8Array(this._numSplats);
      this._splatStates.fill(0);
      this._needsStateTextureUpdate = true;
    }

    if (this._numSplats === 0) {
      return;
    }

    if (this._needsGaussianSplatTexture) {
      if (!this._gaussianSplatTexturePending) {
        GaussianSplatPrimitive.generateSplatTexture(this, frameState);
        if (defined(this._shData)) {
          const oldTex = this.sphericalHarmonicsTexture;
          const width = ContextLimits.maximumTextureSize;
          const dims =
            tileset._selectedTiles[0].content
              .sphericalHarmonicsCoefficientCount / 3;
          const splatsPerRow = Math.floor(width / dims);
          const floatsPerRow = splatsPerRow * (dims * 2);
          const texBuf = new Uint32Array(
            width * Math.ceil(this._numSplats / splatsPerRow) * 2,
          );

          let dataIndex = 0;
          for (let i = 0; dataIndex < this._shData.length; i += width * 2) {
            texBuf.set(
              this._shData.subarray(dataIndex, dataIndex + floatsPerRow),
              i,
            );
            dataIndex += floatsPerRow;
          }
          this.sphericalHarmonicsTexture = createSphericalHarmonicsTexture(
            frameState.context,
            {
              data: texBuf,
              width: width,
              height: Math.ceil(this._numSplats / splatsPerRow),
            },
          );
          if (defined(oldTex)) {
            oldTex.destroy();
          }
        }
      }
      return;
    }

    // Update state texture if needed (after splat texture is ready)
    if (this._needsStateTextureUpdate || !defined(this._splatStateTexture)) {
      if (!this._updateCheckCount) {
        this._updateCheckCount = 0;
      }
      this._updateCheckCount++;

      if (this._updateCheckCount === 1) {
        console.log("[更新流程] 检查状态纹理更新:");
        console.log(
          "  - needsStateTextureUpdate:",
          this._needsStateTextureUpdate,
        );
        console.log("  - stateTexture 存在:", defined(this._splatStateTexture));
        console.log("  - lastTextureWidth:", this._lastTextureWidth);
      }
      GaussianSplatPrimitive.updateSplatStateTexture(this, frameState);
    }

    Matrix4.clone(frameState.camera.viewMatrix, this._prevViewMatrix);
    Matrix4.multiply(
      frameState.camera.viewMatrix,
      this._rootTransform,
      scratchMatrix4A,
    );

    if (!defined(this._sorterPromise)) {
      this._sorterPromise = GaussianSplatSorter.radixSortIndexes({
        primitive: {
          positions: new Float32Array(this._positions),
          modelView: Float32Array.from(scratchMatrix4A),
          count: this._numSplats,
        },
        sortType: "Index",
      });
    }

    if (!defined(this._sorterPromise)) {
      this._sorterState = GaussianSplatSortingState.WAITING;
      return;
    }
    this._sorterPromise.catch((err) => {
      this._sorterState = GaussianSplatSortingState.ERROR;
      this._sorterError = err;
    });
    this._sorterPromise.then((sortedData) => {
      this._indexes = sortedData;
      this._sorterState = GaussianSplatSortingState.SORTED;
    });
  } else if (this._sorterState === GaussianSplatSortingState.WAITING) {
    if (!defined(this._sorterPromise)) {
      this._sorterPromise = GaussianSplatSorter.radixSortIndexes({
        primitive: {
          positions: new Float32Array(this._positions),
          modelView: Float32Array.from(scratchMatrix4A),
          count: this._numSplats,
        },
        sortType: "Index",
      });
    }
    if (!defined(this._sorterPromise)) {
      this._sorterState = GaussianSplatSortingState.WAITING;
      return;
    }
    this._sorterPromise.catch((err) => {
      this._sorterState = GaussianSplatSortingState.ERROR;
      this._sorterError = err;
    });
    this._sorterPromise.then((sortedData) => {
      this._indexes = sortedData;
      this._sorterState = GaussianSplatSortingState.SORTED;
    });

    this._sorterState = GaussianSplatSortingState.SORTING; //set state to sorting
  } else if (this._sorterState === GaussianSplatSortingState.SORTING) {
    return; //still sorting, wait for next frame
  } else if (this._sorterState === GaussianSplatSortingState.SORTED) {
    //update the draw command if sorted
    GaussianSplatPrimitive.buildGSplatDrawCommand(this, frameState);
    this._sorterState = GaussianSplatSortingState.IDLE; //reset state for next frame
    this._dirty = false;
    this._sorterPromise = undefined; //reset promise for next frame
  } else if (this._sorterState === GaussianSplatSortingState.ERROR) {
    throw this._sorterError;
  }

  this._dirty = false;
};

export default GaussianSplatPrimitive;
