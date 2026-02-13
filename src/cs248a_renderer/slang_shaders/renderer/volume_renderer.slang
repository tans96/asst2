implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.
    Ray ray = uniforms.camera.generateRay(uv);
    Ray localRay = ray.transform(uniforms.volume.invModelMatrix);

    // Convert world-space tMax to local-space tMax.
    // Transform the world-space hit point into local space, then compute
    // the local t by projecting onto the local ray direction.
    float localTMax = float.maxValue;
    if (tMax < float.maxValue) {
        float3 worldHitPoint = ray.origin + tMax * ray.direction;
        float4 localHitPoint4 = mul(uniforms.volume.invModelMatrix, float4(worldHitPoint, 1.0));
        float3 localHitPoint = localHitPoint4.xyz;
        // Project onto the local ray direction to get the local t value
        localTMax = dot(localHitPoint - localRay.origin, localRay.direction) / dot(localRay.direction, localRay.direction);
    }

    // Pass localTMax via tRange so accumulateColor can stop at the primitive
    localRay.tRange = float2(localRay.tRange.x, localTMax);

    BoundingBox volumeBound = uniforms.volume.bound;

    // Perform ray marching and accumulate color
    float4 volumeColor = accumulateColor<4>(localRay, volumeBound, MAX_STEPS, STEP_SIZE, DENSITY_SCALE, primColor, uniforms.volume.tex);
    return volumeColor;
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.

    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.
    float2 tHit = volumeBound.hit(ray);
    if (tHit.y < 0.0 || tHit.x > tHit.y) {
        // Ray misses the volume, return background color.
        return backgroundColor;
    }
    // Clamp far hit to ray tRange (stops at primitive if closer)
    tHit.y = min(tHit.y, ray.tRange.y);

    // Front-to-back compositing: accumulate color and transmittance separately,
    // then apply background at the end.
    float3 accColor = float3(0.0);  // accumulated emission
    float T = 1.0;                   // transmittance (attenuation)
    float t = tHit.x;
    uint step = 0;
    [MaxIters(1024)]
    while (step < maxSteps) {
        t += stepSize;  
        if (t > tHit.y) {
            break; // Exceeded the volume bounds
        }
        float3 pos = ray.origin + t * ray.direction;
        float3 uvw = (pos - volumeBound.pMin) / (volumeBound.pMax - volumeBound.pMin);
        float4 texSample = sampleTrilinear(data, uvw);
        float a = texSample.a * densityScale * stepSize;
        accColor += T * a * texSample.rgb;
        T *= (1.0 - a);
        step++;
    }

    // Composite: volume color in front, background behind with remaining transmittance
    float3 finalRgb = accColor + T * backgroundColor.rgb;
    return float4(finalRgb, 1.0);

    // TODO: Student implementation ends here.
}
