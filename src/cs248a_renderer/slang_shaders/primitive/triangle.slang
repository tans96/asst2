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
        float3 v10 = vertices[1] - vertices[0];
        float3 v20 = vertices[2] - vertices[0];
        float3 vp = p - vertices[0];
        
        float d00 = dot(v10, v10);
        float d01 = dot(v10, v20);
        float d11 = dot(v20, v20);
        float d20 = dot(vp, v10);
        float d21 = dot(vp, v20);
        
        float denom = d00 * d11 - d01 * d01;
        if (abs(denom) < EPSILON)
        {
            // Degenerate triangle, return equal weights
            return float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        }
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;

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
        float epsilon = 1e-7;
        // Ensure normal faces the ray
        if (dot(n, ray.direction) > 0) {
            n = -n;
        }
        float denom = dot(n, ray.direction);
        if (abs(denom) < epsilon)
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
            if (abs(D) < epsilon)  // Degenerate triangle
            {
                return Optional<RayHitResult>();
            }
            float s = (uv * wv - vv * wu) / D;
            if (s < -epsilon || s > 1.0 + epsilon)
            {
                return Optional<RayHitResult>();
            }
            float t_param = (uv * wu - uu * wv) / D;
            if (t_param < -epsilon || (s + t_param) > 1.0 + epsilon)
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
