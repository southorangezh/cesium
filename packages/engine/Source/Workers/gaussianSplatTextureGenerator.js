import createTaskProcessorWorker from "./createTaskProcessorWorker.js";
import defined from "../Core/defined.js";

import { initSync, generate_splat_texture } from "@cesium/wasm-splats";

//load built wasm modules for sorting. Ensure we can load webassembly and we support SIMD.
async function initWorker(parameters, transferableObjects) {
  // Require and compile WebAssembly module, or use fallback if not supported
  const wasmConfig = parameters.webAssemblyConfig;
  if (defined(wasmConfig) && defined(wasmConfig.wasmBinary)) {
    initSync({ module: wasmConfig.wasmBinary });
    return true;
  }
  return false;
}

async function generateSplatTextureWorker(parameters, transferableObjects) {
  const wasmConfig = parameters.webAssemblyConfig;
  if (defined(wasmConfig)) {
    return initWorker(parameters, transferableObjects);
  }

  const { attributes, count } = parameters;

  // ========================================
  // [方案1] 添加调试日志
  // ========================================
  console.log(`\n[方案1-Worker] generateSplatTextureWorker 调用:`);
  console.log(`[方案1-Worker]   - count: ${count}`);
  console.log(`[方案1-Worker]   - positions: ${attributes.positions.length}`);
  console.log(`[方案1-Worker]   - scales: ${attributes.scales.length}`);
  console.log(`[方案1-Worker]   - rotations: ${attributes.rotations.length}`);
  console.log(`[方案1-Worker]   - colors: ${attributes.colors.length}`);
  console.log(
    `[方案1-Worker]   - plyIndices 存在: ${defined(attributes.plyIndices)}`,
  );
  if (defined(attributes.plyIndices)) {
    console.log(
      `[方案1-Worker]   - plyIndices 长度: ${attributes.plyIndices.length}`,
    );
    console.log(
      `[方案1-Worker]   - plyIndices 前5个: [${Array.from(attributes.plyIndices.slice(0, 5)).join(", ")}]`,
    );
  }

  // ========================================
  // [方案1] 调用 WASM 生成原始纹理（不包含 PLY 索引）
  // ========================================
  const result = generate_splat_texture(
    attributes.positions,
    attributes.scales,
    attributes.rotations,
    attributes.colors,
    count,
  );

  console.log(`[方案1-Worker] ✓ WASM 纹理生成完成:`);
  console.log(`[方案1-Worker]   - width: ${result.width}`);
  console.log(`[方案1-Worker]   - height: ${result.height}`);
  console.log(`[方案1-Worker]   - data 长度: ${result.data.length}`);

  // ========================================
  // [方案1] JavaScript 端后处理：插入 PLY 索引到 texel 0
  // ========================================
  if (
    defined(attributes.plyIndices) &&
    attributes.plyIndices.length === count
  ) {
    console.log(`\n[方案1-Worker] 开始后处理：插入 PLY 索引...`);

    // 纹理数据是 Uint32Array，每个 splat 占用 2 个 texel（8 个 Uint32）
    // texel 0: [pos.x, pos.y, pos.z, padding] -> 修改为 [pos.x, pos.y, pos.z, plyIndex]
    // texel 1: [cov data, color]

    const textureData = new Uint32Array(result.data);
    const width = result.width;

    for (let i = 0; i < count; i++) {
      // 计算纹理坐标
      const texelX = (i & 0x3ff) << 1; // i % 1024 * 2
      const texelY = i >> 10; // i / 1024

      // 计算在纹理数据中的索引
      // texel 0 的第 4 个分量（索引 3）
      const dataIndex = (texelY * width + texelX) * 4 + 3;

      // 插入 PLY 索引
      const plyIndex = attributes.plyIndices[i];
      textureData[dataIndex] = plyIndex;

      // 调试：输出前几个
      if (i < 5) {
        console.log(
          `[方案1-Worker]   splat ${i}: texel[${texelX},${texelY}] dataIndex=${dataIndex} plyIndex=${plyIndex}`,
        );
      }
    }

    console.log(`[方案1-Worker] ✓ PLY 索引插入完成！`);
    console.log(`[方案1-Worker]   - 已处理 ${count} 个 splat`);
    console.log(`[方案1-Worker]   - 验证前5个 PLY 索引:`);
    for (let i = 0; i < Math.min(5, count); i++) {
      const texelX = (i & 0x3ff) << 1;
      const texelY = i >> 10;
      const dataIndex = (texelY * width + texelX) * 4 + 3;
      console.log(
        `[方案1-Worker]     splat ${i}: plyIndex=${textureData[dataIndex]} (期望: ${attributes.plyIndices[i]})`,
      );
    }

    result.data = textureData.buffer;
  } else {
    console.warn(
      `[方案1-Worker] ⚠️ 未插入 PLY 索引（plyIndices 不存在或长度不匹配）`,
    );
  }

  return {
    data: result.data,
    width: result.width,
    height: result.height,
  };
}

export default createTaskProcessorWorker(generateSplatTextureWorker);
