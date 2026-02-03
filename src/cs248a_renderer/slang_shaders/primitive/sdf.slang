implementing "../primitive.slang";
import math;

public struct SDFBuffer
{
    public StructuredBuffer<SphereSDF> spheres;
    public StructuredBuffer<CubeSDF> cubes;
    public uint sphereCount;
    public uint cubeCount;
    public CustomSDF customSDF;
    public bool renderCustomSDF;

    /**
     * Compute the minimum signed distance from a point to all SDF primitives in the buffer.
     * @param point The point from which to compute the signed distance.
     * @return A Tuple containing the minimum signed distance and the normal of the closest SDF primitive.
     */
    public Tuple<float, float3> sample(float3 point)
    {
        float minDist = float.maxValue;
        float3 closestNormal = float3(0.0, 0.0, 0.0);
        for (uint i = 0; i < this.sphereCount; i++)
        {
            float dist = this.spheres[i].sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.spheres[i].normal(point);
            }
        }
        for (uint i = 0; i < this.cubeCount; i++)
        {
            float dist = this.cubes[i].sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.cubes[i].normal(point);
            }
        }
        if (this.renderCustomSDF)
        {
            float dist = this.customSDF.sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.customSDF.normal(point);
            }
        }
        return Tuple<float, float3>(minDist, closestNormal);
    }
}

// Compose SDFs together using union, intersection, and difference operations
[Differentiable]
public float sdfUnion(float d1, float d2)
{
    return min(d1, d2);
}

[Differentiable]
public float sdfIntersection(float d1, float d2)
{
    return max(d1, d2);
}

[Differentiable]
public float sdfDifference(float d1, float d2)
{
    return max(d1, -d2);
}

public interface ISDF
{
    /**
     * Compute the signed distance from a po˝int to the surface of the primitive.
     * @param point The point from which to compute the signed distance.
     * @return The signed distance from the point to the surface of the primitive.
     */
    public float sdf(float3 point);

    /**
     * The normal of the SDF at a given point.
     * @param point The point at which to compute the normal.
     * @return The normal vector at the given point.
     */
    public float3 normal(float3 point);
}

public struct SphereSDF : ISDF
{
    public float3 center;
    public float radius;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        // TODO: Student implementation starts here.
        float3 localPoint = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        return distance(localPoint, this.center) - this.radius;

        // TODO: Student implementation ends here.
    }

    public float3 normal(float3 point)
    {
        // TODO: Student implementation starts here.
        
        float3 localPoint = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        float3 localNormal = normalize(localPoint - this.center);
        return normalize(mul(transpose(this.invModelMatrix), float4(localNormal, 0.0)).xyz);

        // TODO: Student implementation ends here.
    }
}

public struct CubeSDF : ISDF
{
    public float3 size;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        // TODO: Student implementation starts here.
        float3 localPoint = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        float3 d = abs(localPoint) - this.size;
        return length(max(d, 0.0)) + min(max(max(d.x, d.y), d.z), 0.0);

        // TODO: Student implementation ends here.
    }

    public float3 normal(float3 point)
    {
        // TODO: Student implementation starts here.

        float3 localPoint = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        float3 d = abs(localPoint) - this.size;
        
        // Find which dominant axis
        float maxDist = max(max(d.x, d.y), d.z);
        
        // Normal points along that axis, in the direction of the point
        float3 localNormal = float3(
            abs(d.x - maxDist) < 0.001 ? sign(localPoint.x) : 0.0,
            abs(d.y - maxDist) < 0.001 ? sign(localPoint.y) : 0.0,
            abs(d.z - maxDist) < 0.001 ? sign(localPoint.z) : 0.0
        );
        
        // Transform to world space
        return normalize(mul(transpose(this.invModelMatrix), float4(localNormal, 0.0)).xyz);

        // TODO: Student implementation ends here.
    }
}

public struct CustomSDF : ISDF
{
    public float3 cubeSize;
    public float sphereRadius;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        float3 localPoint = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        CubeSDF outerCube;
        outerCube.size = this.cubeSize;
        outerCube.invModelMatrix = float4x4(
            float4(1.0, 0.0, 0.0, 0.0),
            float4(0.0, 1.0, 0.0, 0.0),
            float4(0.0, 0.0, 1.0, 0.0),
            float4(0.0, 0.0, 0.0, 1.0)); // Identity matrix
        SphereSDF innerSphere;
        innerSphere.center = float3(0.0, 0.0, 0.0);
        innerSphere.radius = this.sphereRadius;
        innerSphere.invModelMatrix = float4x4(
            float4(1.0, 0.0, 0.0, 0.0),
            float4(0.0, 1.0, 0.0, 0.0),
            float4(0.0, 0.0, 1.0, 0.0),
            float4(0.0, 0.0, 0.0, 1.0)); // Identity matrix
        float outerDist = outerCube.sdf(localPoint);
        float innerDist = innerSphere.sdf(localPoint);
        return sdfDifference(outerDist, innerDist);
    }

    [Differentiable]
    static float sdfHelper(no_diff CustomSDF sdf, float3 point)
    {
        return sdf.sdf(point);
    }

    public float3 normal(float3 point)
    {
        DifferentialPair<float3> dp = diffPair(point);
        bwd_diff(CustomSDF.sdfHelper)(this, dp, 1.0);
        return normalize(dp.d);
    }
}
