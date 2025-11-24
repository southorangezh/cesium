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

const scratchMatrix4A = new Matrix4();
const scratchMatrix4B = new Matrix4();
const scratchMatrix4C = new Matrix4();
const scratchMatrix4D = new Matrix4();
const scratchFilterPosition = new Cartesian3();
const scratchFilterRgba = new Uint8Array(4);
const scratchFilterColor = new Color();

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
});

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
  this._positions = undefined;
  this._rotations = undefined;
  this._scales = undefined;
  this._colors = undefined;
  this._indexes = undefined;
  if (defined(this.gaussianSplatTexture)) {
    this.gaussianSplatTexture.destroy();
    this.gaussianSplatTexture = undefined;
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

  renderResources.instanceCount = primitive._numSplats;
  renderResources.count = 4;
  renderResources.primitiveType = PrimitiveType.TRIANGLE_STRIP;
  shaderBuilder.addVertexLines(GaussianSplatVS);
  shaderBuilder.addFragmentLines(GaussianSplatFS);

  const shaderProgram = shaderBuilder.buildShaderProgram(frameState.context);

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
  });

  primitive._drawCommand = command;
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

  if (this._drawCommand) {
    frameState.commandList.push(this._drawCommand);
  }

  if (tileset._modelMatrixChanged) {
    this._dirty = true;
    return;
  }

  if (frameState.passes.pick === true) {
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
