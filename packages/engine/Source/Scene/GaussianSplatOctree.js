import Cartesian3 from "../Core/Cartesian3.js";
import defined from "../Core/defined.js";

function intersectsFrustumAabb(planes, minX, minY, minZ, maxX, maxY, maxZ) {
  // Plane format: Cartesian4 (nx, ny, nz, d), inside test: n·p + d >= 0.
  for (let i = 0; i < planes.length; i++) {
    const plane = planes[i];
    const nx = plane.x;
    const ny = plane.y;
    const nz = plane.z;
    const d = plane.w;

    // Positive vertex in direction of plane normal.
    const px = nx >= 0.0 ? maxX : minX;
    const py = ny >= 0.0 ? maxY : minY;
    const pz = nz >= 0.0 ? maxZ : minZ;

    if (nx * px + ny * py + nz * pz + d < 0.0) {
      return false;
    }
  }
  return true;
}

function computeBounds(positions, count) {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < count; i++) {
    const base = i * 3;
    const x = positions[base];
    const y = positions[base + 1];
    const z = positions[base + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }

  return { minX, minY, minZ, maxX, maxY, maxZ };
}

class GaussianSplatOctreeNode {
  constructor(minX, minY, minZ, maxX, maxY, maxZ, indices, depth) {
    this.minX = minX;
    this.minY = minY;
    this.minZ = minZ;
    this.maxX = maxX;
    this.maxY = maxY;
    this.maxZ = maxZ;
    this.depth = depth;
    this.indices = indices; // Uint32Array for leaf, undefined for internal
    this.children = undefined; // Array(8) of nodes
  }

  get isLeaf() {
    return defined(this.indices);
  }
}

/**
 * Octree over splat positions in root space.
 * Designed for fast candidate pruning for picking.
 *
 * Options:
 * - leafCapacity: max points per leaf
 * - maxDepth: max subdivision depth
 */
export default class GaussianSplatOctree {
  constructor(positionsRootSpace, count, options) {
    this._positions = positionsRootSpace;
    this._count = count;
    this._leafCapacity = options?.leafCapacity ?? 2048;
    this._maxDepth = options?.maxDepth ?? 12;

    const b = computeBounds(this._positions, this._count);
    this._bounds = b;

    // Build a sequential index list once.
    const rootIndices = new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      rootIndices[i] = i;
    }

    this._root = this._buildNode(
      b.minX,
      b.minY,
      b.minZ,
      b.maxX,
      b.maxY,
      b.maxZ,
      rootIndices,
      0,
    );
  }

  destroy() {
    this._root = undefined;
    this._positions = undefined;
    this._count = 0;
  }

  _buildNode(minX, minY, minZ, maxX, maxY, maxZ, indices, depth) {
    if (indices.length <= this._leafCapacity || depth >= this._maxDepth) {
      return new GaussianSplatOctreeNode(
        minX,
        minY,
        minZ,
        maxX,
        maxY,
        maxZ,
        indices,
        depth,
      );
    }

    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;

    const buckets = [[], [], [], [], [], [], [], []];
    const positions = this._positions;

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const base = idx * 3;
      const x = positions[base];
      const y = positions[base + 1];
      const z = positions[base + 2];
      const ox = x > cx ? 1 : 0;
      const oy = y > cy ? 2 : 0;
      const oz = z > cz ? 4 : 0;
      buckets[ox | oy | oz].push(idx);
    }

    // If split failed (all points in one bucket), stop subdividing.
    let nonEmpty = 0;
    for (let i = 0; i < 8; i++) {
      if (buckets[i].length > 0) nonEmpty++;
    }
    if (nonEmpty <= 1) {
      return new GaussianSplatOctreeNode(
        minX,
        minY,
        minZ,
        maxX,
        maxY,
        maxZ,
        indices,
        depth,
      );
    }

    const node = new GaussianSplatOctreeNode(
      minX,
      minY,
      minZ,
      maxX,
      maxY,
      maxZ,
      undefined,
      depth,
    );
    node.children = new Array(8);

    for (let oct = 0; oct < 8; oct++) {
      const list = buckets[oct];
      if (list.length === 0) {
        node.children[oct] = undefined;
        continue;
      }

      const childMinX = (oct & 1) !== 0 ? cx : minX;
      const childMaxX = (oct & 1) !== 0 ? maxX : cx;
      const childMinY = (oct & 2) !== 0 ? cy : minY;
      const childMaxY = (oct & 2) !== 0 ? maxY : cy;
      const childMinZ = (oct & 4) !== 0 ? cz : minZ;
      const childMaxZ = (oct & 4) !== 0 ? maxZ : cz;

      // Convert bucket to Uint32Array for compact storage.
      const childIndices = new Uint32Array(list.length);
      for (let i = 0; i < list.length; i++) {
        childIndices[i] = list[i];
      }

      node.children[oct] = this._buildNode(
        childMinX,
        childMinY,
        childMinZ,
        childMaxX,
        childMaxY,
        childMaxZ,
        childIndices,
        depth + 1,
      );
    }

    return node;
  }

  /**
   * Query candidates within a pick frustum defined by planes (in root space).
   *
   * @param {Cartesian4[]} planes Frustum planes (nx, ny, nz, d). Inside test: n·p + d >= 0.
   * @param {Array<number>} outIndices Output array (will be appended to).
   * @param {number} [maxCandidates=200000] Early-out cap.
   */
  queryFrustum(planes, outIndices, maxCandidates) {
    maxCandidates = maxCandidates ?? 200000;
    if (!defined(this._root) || !defined(planes) || planes.length === 0) {
      return outIndices;
    }

    const stack = [this._root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (!node) continue;

      if (
        !intersectsFrustumAabb(
          planes,
          node.minX,
          node.minY,
          node.minZ,
          node.maxX,
          node.maxY,
          node.maxZ,
        )
      ) {
        continue;
      }

      if (node.isLeaf) {
        const indices = node.indices;
        for (let i = 0; i < indices.length; i++) {
          outIndices.push(indices[i]);
          if (outIndices.length >= maxCandidates) {
            return outIndices;
          }
        }
      } else if (defined(node.children)) {
        // Push children (order doesn't matter).
        const children = node.children;
        for (let i = 0; i < 8; i++) {
          if (children[i]) {
            stack.push(children[i]);
          }
        }
      }
    }

    return outIndices;
  }
}

