module renderer;

import math;
import texture;
import material;
import model;
import primitive;

__include "./renderer/triangle_renderer.slang";
__include "./renderer/volume_renderer.slang";

public struct RendererUniform
{
    // Camera parameters.
    public Camera camera;
    // Ambient color, used when no ray hits a primitive
    public float4 ambientColor;
    // Ray samples per pixel
    public uint sqrtSpp;
    // Primitives.
    public StructuredBuffer<Triangle> triangleBuf;
    public uint triangleCount;
    public SharedTexture3DBuffer<float4> surfaceVolumeTexBuf;
    public StructuredBuffer<Volume> surfaceVolumeBuf;
    public uint surfaceVolumeCount;
    public SDFBuffer sdfBuf;
    // Volumetric data.
    public DiffVolume volume;
    // Acceleration structure.
    public bool useBVH;
    public BVH<Triangle> bvh;
    // Materials.
    public PhysicsBasedMaterialTextureBuf physicsBasedMaterialTextureBuf;
    public StructuredBuffer<PhysicsBasedMaterial> physicsBasedMaterialBuf;
    public uint materialCount;
    // Filtering
    public uint filteringMethod;
    // Debug flags.
    public bool renderDepth;
    public bool renderNormal;
    public bool visualizeBarycentricCoords;
    public bool visualizeTexUV;
    public bool visualizeLevelOfDetail;
    public bool visualizeAlbedo;
}

/**
 * Sample the scene at the given uv coordinate.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the ray sample at the given uv coordinate.
 */
public float4 sample(float2 uv, RendererUniform uniforms)
{
    // Ray mesh intersection.

    Optional<RayMeshSampleResult> rayMeshSampleRes = meshSample(uv, uniforms);

    float4 primColor = uniforms.ambientColor;
    if (rayMeshSampleRes.hasValue)
    {
        primColor = rayMeshSampleRes.value.color;
    }

    if (uniforms.renderDepth)
    {
        // Map depth to [0, 1] range for visualization
        return float4(float3(rayMeshSampleRes.value.t), 1.0);
    }

    if (uniforms.renderNormal)
    {
        // Map normal from [-1, 1] to [0, 1] range for visualization
        return float4(rayMeshSampleRes.value.normal * 0.5 + 0.5, 1.0);
    }

    // Ray volume intersection.
    float tMax = rayMeshSampleRes.hasValue ? rayMeshSampleRes.value.t : float.maxValue;
    float4 volumeColor = volumeSample(uv, uniforms, tMax, primColor);

    return volumeColor;
}

/**
 * The entry point of the renderer.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @return The color of the pixel at the given thread coordinate.
 */
public float4 render(uint2 tid, RendererUniform uniforms)
{
    // TODO: Student implementation starts here.
    float4 finalColor = float4(0.0, 0.0, 0.0, 0.0);
    for (uint i = 0; i < uniforms.sqrtSpp; i++){
        for (uint j = 0; j < uniforms.sqrtSpp; j++)
        {
            // Compute the uv coordinate for the current sub-sample.
            // Note that we add 0.5 to tid to sample at the center of the pixel.
            float2 offset = (float2(i, j) + 0.5) / float(uniforms.sqrtSpp);
            float2 uv = (float2(tid) + offset) / float2(uniforms.camera.canvasSize);
            finalColor += sample(uv, uniforms);
        }
    }

    finalColor /= float(uniforms.sqrtSpp * uniforms.sqrtSpp);
    return finalColor;

    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volume sampling at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void sampleVolumeBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    volumeSampleBwd(uv, uniforms, outGrad);
}

/**
 * Backward pass for volume rendering at a pixel.
 * @param tid The 2D coordinate of the current thread. Ranges from (0, 0) to (canvasSize.x - 1, canvasSize.y - 1).
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the rendered pixel color.
 */
public void renderVolumeBwd(uint2 tid, RendererUniform uniforms, float4 outGrad)
{
    // For simplicity, we're not implementing supersampling in the backward pass.
    float2 uv = (float2(tid) + 0.5) / float2(uniforms.camera.canvasSize);
    sampleVolumeBwd(uv, uniforms, outGrad);
}
