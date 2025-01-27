void renderer_change_cloth_color(py::array_t<float> color)
{
    auto ptr_color = (float *)color.request().ptr;
    g_colors[3] = Colour(ptr_color[0], ptr_color[1], ptr_color[2]);
    g_colors[4] = Colour(ptr_color[0], ptr_color[1], ptr_color[2]);
}

pair<int, int> renderer_add_cloth_square(py::array_t<float> position, py::array_t<float> orientation, py::array_t<int> dimension, py::array_t<float> stiffness, float mass, bool center)
{
    // radius
    float radius = g_params.radius;

    // position: float (initX, initY, initZ)
    auto ptr_position = (float *)position.request().ptr;
    Point3 p = Point3(ptr_position[0], ptr_position[1], ptr_position[2]);

    // orientation: float (yaw, pitch, roll) ==> (y, z, x)
    auto ptr_orientation = (float *)orientation.request().ptr;
    Rotation r = Rotation(ptr_orientation[0], ptr_orientation[1], ptr_orientation[2]);
    Mat44 transformation_mat = TransformMatrix(r, p);

    // dimension: int (dimX, dimZ)
    auto ptr_dimension = (int *)dimension.request().ptr;
    int dimX = ptr_dimension[0];
    int dimZ = ptr_dimension[1];
    int num_verts = dimX * dimZ;

    // center
    // true:  position is the object center coordinate.
    // false: position is the object lower corner.
    Vec3 offset = Vec3(0.0, 0.0, 0.0);
    if (center)
        offset = radius * Vec3(dimX - 1, 0.0, dimZ - 1) / 2.0;

    // stiffness: float (stretch, bend, shear)
    auto ptr_stiffness = (float *)stiffness.request().ptr;
    float stretchStiffness = ptr_stiffness[0];
    float bendStiffness = ptr_stiffness[1];
    float shearStiffness = ptr_stiffness[2];

    int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
    float invMass = (dimX * dimZ) / mass;
    Vec3 velocity = Vec3(0.0, 0.0, 0.0);
    int baseIndex = NvFlexGetActiveCount(g_solver);

    MapBuffers(g_buffers);

    for (int y = 0; y < dimZ; ++y)
    {
        for (int x = 0; x < dimX; ++x)
        {
            int index = baseIndex + y * dimX + x;
            Vec3 position = radius * Vec3(float(x), 0, float(y)) - offset;
            Point3 position_point3 = Point3(position.x, position.y, position.z);
            position_point3 = Multiply(transformation_mat, position_point3);

            g_buffers->positions[index] = Vec4(position_point3.x, position_point3.y, position_point3.z, invMass);
            g_buffers->restPositions[index] = g_buffers->positions[index];
            g_buffers->velocities[index] = velocity;
            g_buffers->phases[index] = phase;
            g_buffers->activeIndices.push_back(index);

            if (x > 0 && y > 0)
            {
                g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dimX));
                g_buffers->triangles.push_back(baseIndex + GridIndex(x, y - 1, dimX));
                g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dimX));

                g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y - 1, dimX));
                g_buffers->triangles.push_back(baseIndex + GridIndex(x, y, dimX));
                g_buffers->triangles.push_back(baseIndex + GridIndex(x - 1, y, dimX));

                g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
                g_buffers->triangleNormals.push_back(Vec3(0.0f, 1.0f, 0.0f));
            }
        }
    }

    // horizontal
    for (int y = 0; y < dimZ; ++y)
    {
        for (int x = 0; x < dimX; ++x)
        {
            int index0 = y * dimX + x;
            if (x > 0)
            {
                int index1 = y * dimX + x - 1;
                CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
            }
            if (x > 1)
            {
                int index2 = y * dimX + x - 2;
                CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
            }
            if (y > 0 && x < dimX - 1)
            {
                int indexDiag = (y - 1) * dimX + x + 1;
                CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
            }

            if (y > 0 && x > 0)
            {
                int indexDiag = (y - 1) * dimX + x - 1;
                CreateSpring(baseIndex + index0, baseIndex + indexDiag, shearStiffness);
            }
        }
    }

    // vertical
    for (int x = 0; x < dimX; ++x)
    {
        for (int y = 0; y < dimZ; ++y)
        {
            int index0 = y * dimX + x;
            if (y > 0)
            {
                int index1 = (y - 1) * dimX + x;
                CreateSpring(baseIndex + index0, baseIndex + index1, stretchStiffness);
            }

            if (y > 1)
            {
                int index2 = (y - 2) * dimX + x;
                CreateSpring(baseIndex + index0, baseIndex + index2, bendStiffness);
            }
        }
    }

    UnmapBuffers(g_buffers);
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, nullptr);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springIndices.size() / 2);
    NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);
    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);
    NvFlexSetActiveCount(g_solver, baseIndex + num_verts);

    return make_pair(baseIndex, num_verts);
}

pair<int, int> renderer_add_cloth_mesh(
    py::array_t<float> position, py::array_t<float> verts, py::array_t<int> faces,
    py::array_t<int> stretch_edges, py::array_t<int> bend_edges,
    py::array_t<int> shear_edges, py::array_t<float> uvs,
    py::array_t<float> stiffness, float mass)
{

    // position: float (initX, initY, initZ)
    auto ptr_position = (float *)position.request().ptr;
    float initX = ptr_position[0];
    float initY = ptr_position[1];
    float initZ = ptr_position[2];
    Vec4 lower = Vec4(initX, initY, initZ, 0);

    // stiffness: float (stretch, bend, shear)
    auto ptr_stiffness = (float *)stiffness.request().ptr;
    float stretchStiffness = ptr_stiffness[0];
    float bendStiffness = ptr_stiffness[1];
    float shearStiffness = ptr_stiffness[2];

    int phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
    int baseIndex = NvFlexGetActiveCount(g_solver);

    MapBuffers(g_buffers);

    // add vertices and uvs
    auto verts_buf = verts.request();
    size_t num_verts = verts_buf.shape[0] / 3;
    auto verts_ptr = (float *)verts_buf.ptr;
    float invMass = num_verts / mass;

    auto uvs_buf = uvs.request();
    assert((bool)(uvs_buf.shape[0] == num_verts));
    auto uvs_ptr = (float *)uvs_buf.ptr;

    for (size_t idx = 0; idx < num_verts; idx++)
    {
        g_buffers->positions[baseIndex + idx] = Vec4(verts_ptr[3 * idx], verts_ptr[3 * idx + 1], verts_ptr[3 * idx + 2], invMass) + lower;
        g_buffers->restPositions[baseIndex + idx] = g_buffers->positions[baseIndex + idx];
        g_buffers->velocities[baseIndex + idx] = Vec3(0, 0, 0);
        g_buffers->phases[baseIndex + idx] = phase;
        g_buffers->activeIndices.push_back(baseIndex + idx);
        g_buffers->uvs[baseIndex + idx] = Vec3(uvs_ptr[3 * idx],
                                               uvs_ptr[3 * idx + 1],
                                               uvs_ptr[3 * idx + 2]);
    }

    // add stretch_edges
    auto stretch_edges_buf = stretch_edges.request();
    size_t num_stretch_edges = stretch_edges_buf.shape[0] / 2;
    auto stretch_edges_ptr = (int *)stretch_edges_buf.ptr;
    for (size_t idx = 0; idx < num_stretch_edges; idx++)
        CreateSpring(baseIndex + stretch_edges_ptr[2 * idx], baseIndex + stretch_edges_ptr[2 * idx + 1], stretchStiffness);

    // add bend_edges
    auto bend_edges_buf = bend_edges.request();
    size_t num_bend_edges = bend_edges_buf.shape[0] / 2;
    auto bend_edges_ptr = (int *)bend_edges_buf.ptr;
    for (size_t idx = 0; idx < num_bend_edges; idx++)
        CreateSpring(baseIndex + bend_edges_ptr[2 * idx], baseIndex + bend_edges_ptr[2 * idx + 1], bendStiffness);

    // add shear_edges
    auto shear_edges_buf = shear_edges.request();
    size_t num_shear_edges = shear_edges_buf.shape[0] / 2;
    auto shear_edges_ptr = (int *)shear_edges_buf.ptr;
    for (size_t idx = 0; idx < num_shear_edges; idx++)
        CreateSpring(baseIndex + shear_edges_ptr[2 * idx], baseIndex + shear_edges_ptr[2 * idx + 1], shearStiffness);

    // add faces
    auto faces_buf = faces.request();
    size_t num_faces = faces_buf.shape[0] / 3;
    auto faces_ptr = (int *)faces_buf.ptr;
    for (size_t idx = 0; idx < num_faces; idx++)
    {
        g_buffers->triangles.push_back(baseIndex + faces_ptr[3 * idx]);
        g_buffers->triangles.push_back(baseIndex + faces_ptr[3 * idx + 1]);
        g_buffers->triangles.push_back(baseIndex + faces_ptr[3 * idx + 2]);
        auto p1 = g_buffers->positions[baseIndex + faces_ptr[3 * idx]];
        auto p2 = g_buffers->positions[baseIndex + faces_ptr[3 * idx + 1]];
        auto p3 = g_buffers->positions[baseIndex + faces_ptr[3 * idx + 2]];
        auto U = p2 - p1;
        auto V = p3 - p1;
        auto normal = Vec3(
            U.y * V.z - U.z * V.y,
            U.z * V.x - U.x * V.z,
            U.x * V.y - U.y * V.x);
        g_buffers->triangleNormals.push_back(normal / Length(normal));
    }

    UnmapBuffers(g_buffers);
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, nullptr);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    NvFlexSetSprings(g_solver, g_buffers->springIndices.buffer, g_buffers->springLengths.buffer, g_buffers->springStiffness.buffer, g_buffers->springIndices.size() / 2);
    NvFlexSetDynamicTriangles(g_solver, g_buffers->triangles.buffer, g_buffers->triangleNormals.buffer, g_buffers->triangles.size() / 3);
    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);
    NvFlexSetActiveCount(g_solver, baseIndex + num_verts);

    return make_pair(baseIndex, num_verts);
}

pair<int, int> renderer_emit_particles_cone(py::array_t<float> position, py::array_t<float> orientation, int num_layer, float alpha, float velocity, float mass, int input_baseIndex)
{
    // position: float (initX, initY, initZ)
    auto ptr_position = (float *)position.request().ptr;
    Point3 p = Point3(ptr_position[0], ptr_position[1], ptr_position[2]);

    // orientation: float (yaw, pitch, roll) ==> (y, z, x)
    auto ptr_orientation = (float *)orientation.request().ptr;
    Rotation r = Rotation(ptr_orientation[0], ptr_orientation[1], ptr_orientation[2]);
    Mat44 transformation_mat = TransformMatrix(r, p);
    Mat44 rotation_mat = TransformMatrix(r, Point3(0, 0, 0));

    int phase = NvFlexMakePhase(1, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);
    float invMass = 1.0 / mass;
    int baseIndex = input_baseIndex != -1 ? input_baseIndex : NvFlexGetActiveCount(g_solver);

    MapBuffers(g_buffers);

    const float pi = 3.14159265358979323846;
    int num_particle = 0;

    alpha = alpha / 180.0 * pi;

    for (int layer_id = 0; layer_id <= num_layer; layer_id++)
    {
        float r = layer_id * g_params.radius;
        float beta = layer_id ? (alpha / num_layer * layer_id) : 0;
        float random_angle = 2 * pi * rand() / RAND_MAX;
        int num_particle_per_layer = max(1, layer_id * 6);
        for (int i = 0; i < num_particle_per_layer; i++)
        {
            int index = baseIndex + num_particle + i + 1;
            float angle = 2.0 * pi * i / num_particle_per_layer + random_angle;
            Vec3 position = r * Vec3(cos(angle), 0, sin(angle));
            Point3 position_point3 = Point3(position.x, position.y, position.z);
            position_point3 = Multiply(transformation_mat, position_point3);

            Point3 velocity_point3 = Multiply(rotation_mat, Point3(sin(beta) * cos(angle), cos(beta), sin(beta) * sin(angle)));
            Vec3 velocity_vec3 = velocity * Vec3(velocity_point3.x, velocity_point3.y, velocity_point3.z);

            g_buffers->positions[index] = Vec4(position_point3.x, position_point3.y, position_point3.z, invMass);
            g_buffers->restPositions[index] = g_buffers->positions[index];
            g_buffers->velocities[index] = velocity_vec3;
            g_buffers->phases[index] = phase;
            if (input_baseIndex == -1)
                g_buffers->activeIndices.push_back(index);
        }
        num_particle += num_particle_per_layer;
    }

    UnmapBuffers(g_buffers);
    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);
    if (input_baseIndex == -1)
        NvFlexSetActiveCount(g_solver, baseIndex + num_particle);
    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, nullptr);
    NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    return make_pair(baseIndex, num_particle);
}