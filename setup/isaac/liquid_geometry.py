"""Deterministic closed review meshes, not a fluid dynamics simulation.

Standard-library-only until Kit starts. All dimensions are meters. Body and
stream are independent meshes with persistent semantic parent identities.
"""
import math


def body_mesh(row, radius, envelope_height):
    n, rings = 64, 10
    height = max(row['level']*envelope_height, .00008)
    footprint = radius*row.get('pool_radius_fraction', .96)
    meniscus = min(row.get('meniscus_m', .0012), height*.6)
    ripple = min(row.get('ripple_m', .00035), height*.35)
    phase = row.get('phase', 0.)

    def surface(r, theta):
        # Raised contact line, a shallow impact depression and radial ripples.
        raised = meniscus*r**10
        waves = ripple*(math.sin(14*r+phase)+.35*math.sin(3*theta+phase))*r*(1-r)
        depression = min(height*.25, .0005)*math.exp(-35*r*r) if row['stream']=='continuous' else 0.
        return max(.00004, height+raised+waves-depression)

    points = [(0., 0., surface(0., 0.))]
    faces = []
    for k in range(1, rings+1):
        r = k/rings
        for j in range(n):
            t = 2*math.pi*j/n
            irregular = 1+.012*math.sin(3*t+phase)*(1-row.get('pool_radius_fraction', .96))
            rr = footprint*r*irregular
            points.append((rr*math.cos(t), rr*math.sin(t), surface(r, t)))
    for j in range(n):
        faces.append((0, 1+j, 1+(j+1)%n))
    for k in range(rings-1):
        inner, outer = 1+k*n, 1+(k+1)*n
        for j in range(n):
            jj = (j+1)%n
            faces.extend([(inner+j, outer+j, outer+jj), (inner+j, outer+jj, inner+jj)])
    surface_faces=len(faces)
    top, bottom = 1+(rings-1)*n, len(points)
    for j in range(n):
        x, y, _ = points[top+j]
        points.append((x*.995, y*.995, 0.))
    center = len(points)
    points.append((0., 0., 0.))
    for j in range(n):
        jj = (j+1)%n
        faces.extend([(top+j, bottom+j, bottom+jj), (top+j, bottom+jj, top+jj)])
    for j in range(n):
        faces.append((center,bottom+(j+1)%n,bottom+j))
    return points, faces, dict(height_m=height, footprint_radius_m=footprint,
        meniscus_m=meniscus, ripple_m=ripple, model='closed-pool-meniscus-mesh-v2',
        normal_group_faces=[surface_faces,2*n,n])


def stream_mesh(row, nozzle_z, surface_z):
    n, rings = 24, 90
    mode = row['stream']
    radius = row.get('stream_radius_m', .001)
    phase = row.get('phase', 0.)
    if mode == 'drop':
        # A closed pendant drop just below the nozzle; no body target.
        top, bottom = nozzle_z-.0002, nozzle_z-.005
    else:
        top = nozzle_z
        bottom = nozzle_z-row.get('jet_length_fraction', .06)*(nozzle_z-surface_z) if mode=='short' else surface_z
    length = max(top-bottom, .0001)
    points, faces = [], []
    for k in range(rings):
        t = k/(rings-1)
        if mode == 'drop':
            rr = radius*max(.01, math.sin(math.pi*t))*(.65+.6*t)
            wobble = 0.
        else:
            rr = radius*(1+.14*math.sin(31*t+phase)+.06*math.sin(73*t+phase))
            wobble = row.get('jet_wobble_m', .00035)*math.sin(19*t+phase)*math.sin(math.pi*t)
        for j in range(n):
            a = 2*math.pi*j/n
            points.append((wobble+rr*math.cos(a), .4*wobble+rr*math.sin(a), top-t*length))
    for k in range(rings-1):
        for j in range(n):
            jj = (j+1)%n; a=k*n; b=(k+1)*n
            faces.extend([(a+j,b+j,b+jj),(a+j,b+jj,a+jj)])
    top_center, bottom_center = len(points), len(points)+1
    points.extend([(0.,0.,top),(0.,0.,bottom)])
    for j in range(n):
        jj=(j+1)%n
        faces.extend([(top_center,j,jj),(bottom_center,(rings-1)*n+jj,(rings-1)*n+j)])
    return points, faces, dict(model='closed-wavy-jet-or-drop-v2', bottom_z_m=bottom,
        top_z_m=top, nominal_radius_m=radius)


def set_mesh(mesh, points, faces, normal_groups=None):
    """Called after SimulationApp. Author smooth normals and explicit bounds."""
    import numpy as np
    from pxr import Gf, UsdGeom
    p = np.asarray(points, dtype=float); f = np.asarray(faces, dtype=int)
    cross = np.cross(p[f[:,1]]-p[f[:,0]], p[f[:,2]]-p[f[:,0]])
    corner_normals=np.empty((len(f),3,3))
    offset=0
    for count in normal_groups or [len(f)]:
        group=f[offset:offset+count]; normals=np.zeros_like(p)
        for j in range(3): np.add.at(normals,group[:,j],cross[offset:offset+count])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
        corner_normals[offset:offset+count]=normals[group]; offset+=count
    if offset!=len(f): raise ValueError('Normal groups do not cover the closed mesh')
    mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for v in p])
    mesh.GetFaceVertexCountsAttr().Set([3]*len(f))
    mesh.GetFaceVertexIndicesAttr().Set(f.ravel().tolist())
    # Explicit corner normals avoid a transient Hydra/Fabric interpretation of
    # the changing point-sized buffer as face-varying on pinned Isaac 6.0.
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.GetNormalsAttr().Set([Gf.Vec3f(*v) for v in corner_normals.reshape(-1,3)])
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.GetExtentAttr().Set([Gf.Vec3f(*p.min(0)), Gf.Vec3f(*p.max(0))])


def transmission_material(stage, name):
    from pxr import Sdf, UsdShade
    import carb.tokens
    from pathlib import Path
    mdl=Path(carb.tokens.get_tokens_interface().resolve('${kit}/mdl/core/mdl/OmniSurface/OmniSurfaceBase.mdl'))
    if not mdl.is_file():
        raise FileNotFoundError('Pinned OmniSurfaceBase MDL not found: '+str(mdl))
    path = '/SyntheticReview/Materials/'+name
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Surface')
    shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(mdl)), 'mdl')
    shader.SetSourceAssetSubIdentifier('OmniSurfaceBase', 'mdl')
    mat.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(), 'out')
    for key, value in [('diffuse_reflection_weight',0.),('specular_reflection_weight',1.),
                       ('specular_reflection_roughness',.035),('specular_reflection_ior',1.333),
                       ('specular_transmission_weight',1.),('geometry_opacity',1.)]:
        shader.CreateInput(key,Sdf.ValueTypeNames.Float).Set(value)
    for key, value in [('enable_specular_transmission',True),('thin_walled',False),('enable_opacity',False)]:
        shader.CreateInput(key,Sdf.ValueTypeNames.Bool).Set(value)
    return mat, shader


def set_transmission(shader, row, stage_meters):
    from pxr import Gf, Sdf
    shader.CreateInput('specular_transmission_color',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*row['transmission_color']))
    shader.CreateInput('specular_transmission_scattering_depth',Sdf.ValueTypeNames.Float).Set(row['absorption_depth_m']/stage_meters)
    shader.GetInput('specular_reflection_roughness').Set(row.get('roughness',.035))
