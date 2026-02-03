implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.
        // Transform ray to local space
        Ray localRay = ray.transform(invModelMatrix);
        
        // Find ray-box intersection to get entry point
        float2 tMinMax = bound.hit(localRay);
        float tMin = max(tMinMax.x, 0.0);  // Don't start behind the camera
        float tMax = tMinMax.y;
        
        if (tMin > tMax) {
            return Optional<RayHitResult>();  // Ray misses the box
        }
        
        float3 dir = normalize(localRay.direction);
        float3 local_point = localRay.origin + dir * tMin;  // Start at box entry
        float distance = tMin;
        
        while (distance <= tMax) {
            float3 uvw = getUVW(local_point);
            float4 density = volumeTexBuf.trilinearSample(tex, uvw);
            if (density.w >= DENSITY_THRESHOLD) {
                RayHitResult result;
                // Transform back to world space to get proper distance
                float4 worldHitPoint4 = mul(modelMatrix, float4(local_point, 1.0));
                float3 worldHitPoint = worldHitPoint4.xyz / worldHitPoint4.w;
                result.t = length(worldHitPoint - ray.origin);
                // Estimate normal using finite difference.
                float dx = volumeTexBuf.trilinearSample(tex, getUVW(local_point + float3(FINITE_DIFF_DELTA, 0.0, 0.0))).w -
                           volumeTexBuf.trilinearSample(tex, getUVW(local_point - float3(FINITE_DIFF_DELTA, 0.0, 0.0))).w;
                float dy = volumeTexBuf.trilinearSample(tex, getUVW(local_point + float3(0.0, FINITE_DIFF_DELTA, 0.0))).w -
                           volumeTexBuf.trilinearSample(tex, getUVW(local_point - float3(0.0, FINITE_DIFF_DELTA, 0.0))).w;
                float dz = volumeTexBuf.trilinearSample(tex, getUVW(local_point + float3(0.0, 0.0, FINITE_DIFF_DELTA))).w -
                           volumeTexBuf.trilinearSample(tex, getUVW(local_point - float3(0.0, 0.0, FINITE_DIFF_DELTA))).w;
                // Negative sign: gradient points toward increasing density (inward), normal points outward
                float3 localNormal = -normalize(float3(dx, dy, dz));
                // Transform normal to world space using inverse transpose
                float4 worldNormal4 = mul(transpose(invModelMatrix), float4(localNormal, 0.0));
                result.normal = normalize(worldNormal4.xyz);
                return Optional<RayHitResult>(result);
            }
            local_point += dir * STEP_SIZE;
            distance += STEP_SIZE;
        }
        // TODO: Student implementation ends here.

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in local space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
