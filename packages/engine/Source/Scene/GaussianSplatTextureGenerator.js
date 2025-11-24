import defined from "../Core/defined.js";
import FeatureDetection from "../Core/FeatureDetection.js";
import RuntimeError from "../Core/RuntimeError.js";
import TaskProcessor from "../Core/TaskProcessor.js";

function GaussianSplatTextureGenerator() {}

GaussianSplatTextureGenerator._maxSortingConcurrency = Math.max(
  FeatureDetection.hardwareConcurrency - 1,
  1,
);

GaussianSplatTextureGenerator._textureTaskProcessor = undefined;
GaussianSplatTextureGenerator._taskProcessorReady = false;
GaussianSplatTextureGenerator._error = undefined;
GaussianSplatTextureGenerator._getTextureTaskProcessor = function () {
  if (!defined(GaussianSplatTextureGenerator._textureTaskProcessor)) {
    const processor = new TaskProcessor(
      "gaussianSplatTextureGenerator",
      GaussianSplatTextureGenerator._maxSortingConcurrency,
    );
    processor
      .initWebAssemblyModule({
        wasmBinaryFile: "ThirdParty/wasm_splats_bg.wasm",
      })
      .then(function (result) {
        if (result) {
          GaussianSplatTextureGenerator._taskProcessorReady = true;
        } else {
          GaussianSplatTextureGenerator._error = new RuntimeError(
            "Gaussian splat sorter could not be initialized.",
          );
        }
      })
      .catch((error) => {
        GaussianSplatTextureGenerator._error = error;
      });
    GaussianSplatTextureGenerator._textureTaskProcessor = processor;
  }

  return GaussianSplatTextureGenerator._textureTaskProcessor;
};

GaussianSplatTextureGenerator.generateFromAttributes = function (parameters) {
  const textureTaskProcessor =
    GaussianSplatTextureGenerator._getTextureTaskProcessor();
  if (defined(GaussianSplatTextureGenerator._error)) {
    throw GaussianSplatTextureGenerator._error;
  }

  if (!GaussianSplatTextureGenerator._taskProcessorReady) {
    return;
  }

  const { attributes } = parameters;

  // ========================================
  // [方案1] 添加调试日志
  // ========================================
  console.log(`\n[方案1-TextureGen] generateFromAttributes 调用:`);
  console.log(`[方案1-TextureGen]   - count: ${parameters.count}`);
  console.log(
    `[方案1-TextureGen]   - positions: ${attributes.positions.length}`,
  );
  console.log(`[方案1-TextureGen]   - scales: ${attributes.scales.length}`);
  console.log(
    `[方案1-TextureGen]   - rotations: ${attributes.rotations.length}`,
  );
  console.log(`[方案1-TextureGen]   - colors: ${attributes.colors.length}`);
  console.log(
    `[方案1-TextureGen]   - plyIndices 存在: ${defined(attributes.plyIndices)}`,
  );
  if (defined(attributes.plyIndices)) {
    console.log(
      `[方案1-TextureGen]   - plyIndices 长度: ${attributes.plyIndices.length}`,
    );
    console.log(
      `[方案1-TextureGen]   - plyIndices 前5个: [${Array.from(attributes.plyIndices.slice(0, 5)).join(", ")}]`,
    );
  }

  // ========================================
  // [方案1] 传递 PLY 索引缓冲区
  // ========================================
  const transferList = [
    attributes.positions.buffer,
    attributes.scales.buffer,
    attributes.rotations.buffer,
    attributes.colors.buffer,
  ];

  if (defined(attributes.plyIndices)) {
    transferList.push(attributes.plyIndices.buffer);
    console.log(`[方案1-TextureGen] ✓ PLY 索引缓冲区已添加到传输列表`);
  }

  return textureTaskProcessor.scheduleTask(parameters, transferList);
};

export default GaussianSplatTextureGenerator;
