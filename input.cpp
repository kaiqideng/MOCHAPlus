#include "input.h"

void loadTriangleWallInfo(const std::string& file, HostTriangleWall& TW)
{
    std::ifstream fin(file);
    if (!fin) { std::cerr << "Cannot open " << file << '\n'; return; }
    std::string line, key;

    //-----------------  Walls -----------------
    if (!getlineValid(fin, line)) return;
    if (countValidNumbersStrict(line) != 1)
    {
        std::cerr << "Invalid WALLS header format: " << line << '\n';
        return;
    }
    int nW = 0;
    std::istringstream w0(line);
    w0 >> key >> nW;
    if (key != "WALLS")
    {
        std::cerr << "Need 'WALLS' header\n";
        return;
    }
    if (nW <= 0)
    {
        std::cerr << "Number of walls must be positive\n";
        return;
    }
	std::vector<int> materialIndex(nW);
    HostDynamicState state = HostDynamicState(nW);
    for (int i = 0; i < nW; ++i)
    {
        if (!getlineValid(fin, line))
        {
            std::cerr << "Missing wall lines\n";
            return;
        }
        if (countValidNumbersStrict(line) != 9)
        {
            std::cerr << "Invalid wall line format: " << line << '\n';
            return;
        }
        int id = 0, mat = 0; double3 pos = make_double3(0, 0, 0); quaternion q = make_quaternion(1, 0, 0, 0);
        std::istringstream wl(line);
        wl >> id >> mat
            >> pos.x >> pos.y >> pos.z
            >> q.q0 >> q.q1 >> q.q2 >> q.q3;
        if (id < 0 || id >= nW)
        {
            std::cerr << "Wall ID out of range: " << id << '\n';
            return;
        }
        materialIndex[id] = mat;
        state.positions[id] = pos;
        state.orientations[id] = q;
    }

    //-----------------  Vertices -----------------
    if (!getlineValid(fin, line)) return;
    if (countValidNumbersStrict(line) != 1)
    {
        std::cerr << "Invalid VERTICES header format: " << line << '\n';
        return;
    }
    int nV = 0;
    std::istringstream v0(line);
    v0 >> key >> nV;            // key == "VERTICES"
    if (key != "VERTICES")
    {
        std::cerr << "Need 'VERTICES' header\n";
        return;
    }
    if (nV <= 0)
    {
        std::cerr << "Number of vertices must be positive\n";
        return;
    }
    HostTriangleVertex vertex = HostTriangleVertex(nV);
    for (int v = 0; v < nV; ++v)
    {
        if (!getlineValid(fin, line)) return;
        if (countValidNumbersStrict(line) != 4)
        {
            std::cerr << "Invalid vertex line format: " << line << '\n';
            return;
        }
        std::istringstream vl(line);
        vl >> vertex.positions[v].x
            >> vertex.positions[v].y
            >> vertex.positions[v].z
            >> vertex.vertex2Wall[v];
    }

    //-----------------  Faces -----------------
    if (!getlineValid(fin, line)) return;
    if (countValidNumbersStrict(line) != 1)
    {
        std::cerr << "Invalid FACES header format: " << line << '\n';
        return;
    }
    int nF = 0;
    std::istringstream f0(line);
    f0 >> key >> nF;            // key == "FACES"
    if (key != "FACES")
    {
        std::cerr << "Need 'FACES' header\n";
        return;
    }
    if (nF <= 0)
    {
        std::cerr << "Number of faces must be positive\n";
        return;
    }
    HostTriangleFace face = HostTriangleFace(nF);
    for (int f = 0; f < nF; ++f)
    {
        if (!getlineValid(fin, line)) return;
        if (countValidNumbersStrict(line) != 4)
        {
            std::cerr << "Invalid face line format: " << line << '\n';
            return;
        }
        std::istringstream fl(line);
        fl >> face.vAIndex[f]
            >> face.vBIndex[f]
            >> face.vCIndex[f]
            >> face.face2Wall[f];
    }

    //-----------------  Edges -----------------
	TW.num += nW;
    TW.materialIndex.insert(TW.materialIndex.end(), materialIndex.begin(), materialIndex.end());
	TW.state.insertData(state);
	TW.face.num += nF;
	TW.face.vAIndex.insert(TW.face.vAIndex.end(), face.vAIndex.begin(), face.vAIndex.end());
	TW.face.vBIndex.insert(TW.face.vBIndex.end(), face.vBIndex.begin(), face.vBIndex.end());
	TW.face.vCIndex.insert(TW.face.vCIndex.end(), face.vCIndex.begin(), face.vCIndex.end());
	TW.face.face2Wall.insert(TW.face.face2Wall.end(), face.face2Wall.begin(), face.face2Wall.end());
	TW.vertex.num += nV;
	TW.vertex.positions.insert(TW.vertex.positions.end(), vertex.positions.begin(), vertex.positions.end());
	TW.vertex.vertex2Wall.insert(TW.vertex.vertex2Wall.end(), vertex.vertex2Wall.begin(), vertex.vertex2Wall.end());
	TW.vertex.facePrefixSum.insert(TW.vertex.facePrefixSum.end(), vertex.facePrefixSum.begin(), vertex.facePrefixSum.end());
	TW.vertex.edgePrefixSum.insert(TW.vertex.edgePrefixSum.end(), vertex.edgePrefixSum.begin(), vertex.edgePrefixSum.end());
    TW.loadEdgeInfo();
    return;
}

void loadContactParameterInfo(const std::string& file, HostContactParameter& CP)
{
    std::ifstream fin(file);
    if (!fin) { std::cerr << "cannot open " << file << '\n'; return; }
    std::string line, key;
    if (!getlineValid(fin, line)) return;
    if (countValidNumbersStrict(line) != 1)
    {
        std::cerr << "Invalid header format: " << line << '\n';
        return;
    }
    std::istringstream head0(line);
    int m = 0;
    head0 >> key >> m;
    CP = HostContactParameter(m);
    /* -------------------------------------------------- */
    /*  MATERIAL                                       */
    /* -------------------------------------------------- */
    if (key == "MATERIAL")
    {
        CP.material.num = m;
        CP.material.elasticModulus.resize(m, 0.);
        CP.material.poissonRatio.resize(m, 0.);

        for (int k = 0; k < m; ++k)
        {
            getlineValid(fin, line);
            std::istringstream is(line);
            if (countValidNumbersStrict(line) != 3)
            {
                std::cerr << "Invalid material line format: " << line << '\n';
                return;
            }
            int id = 0;
            is >> id
                >> CP.material.elasticModulus[id]
                >> CP.material.poissonRatio[id];
        }
    }
    else
    {
        std::cerr << "Need 'MATERIAL' header firstly\n";
        return;
    }

    while (getlineValid(fin, line))
    {
        std::istringstream head1(line);
        head1 >> key;
        /* -------------------------------------------------- */
        /*  1. HERTZIAN                                       */
        /* -------------------------------------------------- */
        if (key == "HERTZIAN")
        {
            while (getlineValid(fin, line) && line != "END")
            {
                if (countValidNumbersStrict(line) != 8)
                {
                    std::cerr << "Invalid Hertzian line format: " << line << '\n';
                    return;
                }
                std::istringstream is(line);
                int i, j; double rs, ts, res, mus, mur, mut;
                is >> i >> j >> rs >> ts >> res >> mus >> mur >> mut;
                if (i >= CP.material.num || j >= CP.material.num)
                {
                    std::cerr << "Hertzian Contact: Material index out of range" << '\n';
                    return;
                }
                int idx = CP.getContactParameterIndex(i, j);
                CP.Hertzian.kR_to_kS_ratio[idx] = rs;
                CP.Hertzian.kT_to_kS_ratio[idx] = ts;
                CP.Hertzian.restitution[idx] = res;
                CP.Hertzian.friction.sliding[idx] = mus;
                CP.Hertzian.friction.rolling[idx] = mur;
                CP.Hertzian.friction.torsion[idx] = mut;
            }
        }

        /* -------------------------------------------------- */
        /*  2. LINEAR  (kR/kT  cR/cT)                      */
        /* -------------------------------------------------- */
        else if (key == "LINEAR")
        {
            while (getlineValid(fin, line) && line != "END")
            {
                if (countValidNumbersStrict(line) != 13)
                {
                    std::cerr << "Invalid Linear line format: " << line << '\n';
                    return;
                }
                std::istringstream is(line);
                int i, j; double kN, kS, kR, kT, cN, cS, cR, cT, mus, mur, mut;
                is >> i >> j
                    >> kN >> kS >> kR >> kT
                    >> cN >> cS >> cR >> cT
                    >> mus >> mur >> mut;
                if (i >= CP.material.num || j >= CP.material.num)
                {
                    std::cerr << "Linear Contact: Material index out of range" << '\n';
                    return;
                }
                int idx = CP.getContactParameterIndex(i, j);
                CP.Linear.stiffness.normal[idx] = kN;
                CP.Linear.stiffness.sliding[idx] = kS;
                CP.Linear.stiffness.rolling[idx] = kR;
                CP.Linear.stiffness.torsion[idx] = kT;

                CP.Linear.dissipation.normal[idx] = cN;
                CP.Linear.dissipation.sliding[idx] = cS;
                CP.Linear.dissipation.rolling[idx] = cR;
                CP.Linear.dissipation.torsion[idx] = cT;

                CP.Linear.friction.sliding[idx] = mus;
                CP.Linear.friction.rolling[idx] = mur;
                CP.Linear.friction.torsion[idx] = mut;
            }
        }

        /* -------------------------------------------------- */
        /*  3. BONDED                                         */
        /* -------------------------------------------------- */
        else if (key == "BONDED")
        {
            while (getlineValid(fin, line) && line != "END")
            {
                if (countValidNumbersStrict(line) != 10)
                {
                    std::cerr << "Invalid Bonded line format: " << line << '\n';
                    return;
                }
                std::istringstream is(line);
                int i, j; double d, mult, E, ratio, tens, coh, mu, gamma;
                is >> i >> j >> d >> mult >> E >> ratio >> tens >> coh >> mu >> gamma;
                if (i >= CP.material.num || j >= CP.material.num)
                {
                    std::cerr << "Bonded Contact: Material index out of range" << '\n';
                    return;
                }
                int idx = CP.getContactParameterIndex(i, j);
                CP.Bond.maxContactGap[idx] = d;
                CP.Bond.multiplier[idx] = mult;
                CP.Bond.elasticModulus[idx] = E;
                CP.Bond.kN_to_kS_ratio[idx] = ratio;
                CP.Bond.tensileStrength[idx] = tens;
                CP.Bond.cohesion[idx] = coh;
                CP.Bond.frictionCoeff[idx] = mu;
                CP.Bond.criticalDamping[idx] = gamma;
            }
        }
    }
}