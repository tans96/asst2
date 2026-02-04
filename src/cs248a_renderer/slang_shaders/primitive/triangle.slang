implementing "../primitive.slang";
import math;

public struct Triangle : IBoundingBox, IRayCastable
{
    public float3[3] vertices;
    public float3[3] normals;
    public float2[3] uvs;
    public float3[3] colors;
    public uint materialId;
    public uint offset;

    public BoundingBox boundingBox()
    {
        return BoundingBox(float3(0.0), float3(0.0));
    }

    /* Returns the average color of the triangle. */
    public float3 averageColor()
    {
        return (this.colors[0] + this.colors[1] + this.colors[2]) / 3.0;
    }

    /**
     * Calculates the barycentric coordinates of a point on the triangle.
     * @param p The point on the triangle to calculate the barycentric coordinates for.
     * @return The barycentric coordinates of the point.
     */
    public float3 calculateBarycentricCoord(float3 p)
    {
        const float EPSILON = 1e-6;
        float u = 0.0;
        float v = 0.0;
        float w = 0.0;

        // TODO: Student implementation starts here.

        // TODO: Student implementation ends here.

        return float3(u, v, w);
    }

    public float2 getUV(float3 barycentricCoord)
    {
        float2 uv = float2(0.0);

        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        uv = u * this.uvs[0] + v * this.uvs[1] + w * this.uvs[2];
        
        return uv;
    }

    public float3 getColor(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 color = u * this.colors[0] + v * this.colors[1] + w * this.colors[2];
        return color;
    }

    public float3 getNormal(float3 barycentricCoord)
    {
        float u = barycentricCoord.x;
        float v = barycentricCoord.y;
        float w = barycentricCoord.z;
        float3 normal = u * this.normals[0] + v * this.normals[1] + w * this.normals[2];
        return normalize(normal);
    }

    /**
     * Test ray-triangle intersection.
     * @param ray The ray to test against the triangle.
     * @return An Optional containing RayHitResult if the ray hits the triangle, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray)
    {
        // TODO: Student implementation starts here.
        float3 u = vertices[1] - vertices[0];
        float3 v = vertices[2] - vertices[0];
        float3 n = normalize(cross(u, v));
        // Ensure normal faces the ray
        if (dot(n, ray.direction) > 0) {
            n = -n;
        }
        float denom = dot(n, ray.direction);
        if (abs(denom) < 1e-6)
        {
            return Optional<RayHitResult>();
        }
        float c = dot(n, vertices[0]);
        float t = (c - dot(n, ray.origin)) / denom;
        if (t < ray.tRange.x || t > ray.tRange.y)
        {
            return Optional<RayHitResult>();
        } else {
            float3 P = ray.origin + t * ray.direction;
            float3 w = P - vertices[0];
            float uu = dot(u, u);
            float uv = dot(u, v);
            float vv = dot(v, v);
            float wu = dot(w, u);
            float wv = dot(w, v);
            float D = uv * uv - uu * vv;
            if (abs(D) < 1e-10)  // Degenerate triangle
            {
                return Optional<RayHitResult>();
            }
            float s = (uv * wv - vv * wu) / D;
            if (s < 0.0 || s > 1.0)
            {
                return Optional<RayHitResult>();
            }
            float t_param = (uv * wu - uu * wv) / D;
            if (t_param < 0.0 || (s + t_param) > 1.0)
            {
                return Optional<RayHitResult>();
            }
            RayHitResult result;
            result.t = t;
            result.normal = n;
            return Optional<RayHitResult>(result);
        }

        // TODO: Student implementation ends here.
    }

    public uint getMaterialId()
    {
        return this.materialId + this.offset;
    }
}
