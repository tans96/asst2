implementing "../renderer.slang";


float3 simpleShade(float3 rayDir, float3 normal, float3 albedo)
{
    float NdotL = max(dot(normalize(normal), normalize(-rayDir)), 0.0);
    return albedo * NdotL;
}

public struct RayMeshSampleResult
{
    public float t;
    public float3 normal;
    public float4 color;
}

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<RayMeshSampleResult> meshSample(float2 uv, RendererUniform uniforms)
{
    // We generate a camera ray for the given uv coordinate.
    // You need to implement the `generateRay` function in `Camera` struct in `src/cs248a_renderer/slang_shaders/model/camera.slang`
    Ray ray = uniforms.camera.generateRay(uv);
    RayHitResult result;

    Optional<RayMeshSampleResult> allResult;
    // Ray mesh intersection test.
    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult.hasValue)
    {
        RayHitResult result = rayMeshIntersectionResult.value.closestHit;
        float3 hitPoint = ray.origin + result.t * ray.direction;
        float3 barycentricCoord = rayMeshIntersectionResult.value.hitTriangle.calculateBarycentricCoord(hitPoint);
        float3 normal = result.normal;
        float2 tex_uv = rayMeshIntersectionResult.value.hitTriangle.getUV(barycentricCoord);
        float4 color = shadeTriangle(uv, uniforms, rayMeshIntersectionResult.value.hitTriangle, ray, tex_uv, normal, barycentricCoord);
        allResult = Optional<RayMeshSampleResult>(RayMeshSampleResult(result.t, normal, color));
    }

    return allResult;
}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();

    // IMPORTANT CHANGE: We need to store the hit triangle.
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    Optional<Triangle> hitTriangle = Optional<Triangle>();
    // IMPORTANT CHANGE: We need to store the hit triangle.

    // TODO: Student implementation starts here.
    // Naive version: test intersection with all triangles. Set the base normal to be 0,0,0 and t to be max float value.
    float closestT = float.maxValue;
    StructuredBuffer<Triangle> triangles = uniforms.triangleBuf;
    uint triangleCount = uniforms.triangleCount;
    Optional<RayHitResult> closestHit;
    // Sample BVH if enabled
    if (uniforms.useBVH)
    {
        Optional<BVH<Triangle>.HitResult> bvhHit = uniforms.bvh.hit(ray); 
        if (bvhHit.hasValue && bvhHit.value.rayHitResult.t < closestT)
        {
            closestT = bvhHit.value.rayHitResult.t;
            result = bvhHit.value.rayHitResult;
            hitTriangle = bvhHit.value.primitive;
        }
    }
    else
    {
        for (uint i = 0; i < triangleCount; i++)
        {
            Optional<RayHitResult> hitResult = triangles[i].hit(ray);
            if (hitResult.hasValue && hitResult.value.t < closestT)
            {
                closestT = hitResult.value.t;
                result = hitResult.value;
                hitTriangle = triangles[i];
            }
        }

        // Test intersection with volumes
        StructuredBuffer<Volume> volumes = uniforms.surfaceVolumeBuf;
        uint volumeCount = uniforms.surfaceVolumeCount;
        for (uint i = 0; i < volumeCount; i++)
        {
            Optional<RayHitResult> hitResult = volumes[i].hit(ray, uniforms.surfaceVolumeTexBuf);
            if (hitResult.hasValue && hitResult.value.t < closestT)
            {
                closestT = hitResult.value.t;
                result = hitResult.value;
            }
        }

        // Sample SDFs
        float3 currentPoint = ray.origin;
        float distance = float.maxValue;
        for (uint step = 0; step < MAX_STEPS; step++)
        {
            Tuple<float, float3> sdfSample = uniforms.sdfBuf.sample(currentPoint);
            distance = sdfSample._0;
            if (distance < EPSILON)
            {
                float t = length(currentPoint - ray.origin);
                if (t < closestT)
                {
                    closestT = t;
                    result = RayHitResult(t, sdfSample._1);
                }
                break;
            }
            currentPoint += ray.direction * distance;
        }
        
    }

    // TODO: Student implementation ends here.

    if (result.hasValue && hitTriangle.hasValue)
    {
        allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    }

    return allResult;
}

// Compute the level of the texture at the hit point.
// @param uv: The uv coordinates of the hit point.
// @param uniforms: The renderer uniforms.
// @param triangle: The triangle that was hit.
// @param tex_uv: The uv coordinates of the texture at the hit point.
// @return: The level of the texture at the hit point.
public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.
    float dx = 1 / float(uniforms.camera.canvasSize.x);
    float dy = 1 / float(uniforms.camera.canvasSize.y);
    float2 uv_right = uv + float2(dx, 0);
    float2 uv_left = uv - float2(dx, 0);
    float2 uv_up = uv + float2(0, dy);
    float2 uv_down = uv - float2(0, dy);
    Ray ray_right = uniforms.camera.generateRay(uv_right);
    Ray ray_left = uniforms.camera.generateRay(uv_left);
    Ray ray_up = uniforms.camera.generateRay(uv_up);
    Ray ray_down = uniforms.camera.generateRay(uv_down);
    Optional<RayMeshIntersectionResult> rightHit = rayMeshIntersection(ray_right, uniforms);
    Optional<RayMeshIntersectionResult> leftHit = rayMeshIntersection(ray_left, uniforms);
    Optional<RayMeshIntersectionResult> downHit = rayMeshIntersection(ray_down, uniforms);
    Optional<RayMeshIntersectionResult> upHit = rayMeshIntersection(ray_up, uniforms);
    float du_dx = 0.0;
    float dv_dx = 0.0;
    float du_dy = 0.0;
    float dv_dy = 0.0;
    if (rightHit.hasValue && rightHit.value.hitTriangle.getMaterialId() == triangle.getMaterialId() && leftHit.hasValue && leftHit.value.hitTriangle.getMaterialId() == triangle.getMaterialId())
    {
        float3 barycentricCoordRight = rightHit.value.hitTriangle.calculateBarycentricCoord(rightHit.value.closestHit.t * ray_right.direction + ray_right.origin);
        float3 barycentricCoordLeft = leftHit.value.hitTriangle.calculateBarycentricCoord(leftHit.value.closestHit.t * ray_left.direction + ray_left.origin);
        float2 right_tex_uv = rightHit.value.hitTriangle.getUV(barycentricCoordRight);
        float2 left_tex_uv = leftHit.value.hitTriangle.getUV(barycentricCoordLeft);
        float2 duv = right_tex_uv - left_tex_uv;
        du_dx = 0.5 * duv.x / dx;
        dv_dx = 0.5 * duv.y / dx;
    }
    if (upHit.hasValue && upHit.value.hitTriangle.getMaterialId() == triangle.getMaterialId() && downHit.hasValue && downHit.value.hitTriangle.getMaterialId() == triangle.getMaterialId())
    {
        float3 barycentricCoordUp = upHit.value.hitTriangle.calculateBarycentricCoord(upHit.value.closestHit.t * ray_up.direction + ray_up.origin);
        float3 barycentricCoordDown = downHit.value.hitTriangle.calculateBarycentricCoord(downHit.value.closestHit.t * ray_down.direction + ray_down.origin);
        float2 up_tex_uv = upHit.value.hitTriangle.getUV(barycentricCoordUp);
        float2 down_tex_uv = downHit.value.hitTriangle.getUV(barycentricCoordDown);
        float2 duv = up_tex_uv - down_tex_uv;
        du_dy = 0.5 * duv.x / dy;
        dv_dy = 0.5 * duv.y / dy;
    }
    float rho_x = sqrt(du_dx * du_dx + dv_dx * dv_dx);
    float rho_y = sqrt(du_dy * du_dy + dv_dy * dv_dy);
    float rho = max(rho_x, rho_y);
    level = log2(rho);

    return level;

    // TODO: Student implementation ends here.
}

public float4 shadeTriangle(float2 uv, RendererUniform uniforms, Triangle triangle, Ray ray, float2 tex_uv, float3 normal, float3 barycentricCoord)
{
    if (uniforms.visualizeBarycentricCoords)
    {
        return float4(barycentricCoord, 1.0);
    }
    if (uniforms.visualizeTexUV)
    {
        return float4(tex_uv, 0.0, 1.0);
    }
    // Get the level of detail of the texture at the hit point.
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint discreteLevel = uint(level);

    // Sample albedo from the material at the hit point.
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    float3 albedo = parameters.albedo;

    if (uniforms.visualizeLevelOfDetail || uniforms.visualizeAlbedo)
    {
        return float4(float3(albedo), 1.0);
    }

    // Shade the triangle.
    float3 color = simpleShade(ray.direction, normal, albedo);
    return float4(color, 1.0);
}
