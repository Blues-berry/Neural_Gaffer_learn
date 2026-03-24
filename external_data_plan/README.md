# External Data Plan

This folder tracks the phase-1 replacement plan for the two training subsets.
The goal is to rebuild both subsets with external sources as the primary object source, instead of Objaverse.

## Scope

- `ecommerce`: product-photo style objects
- `landscape`: outdoor / nature / sculpture / prop objects for landscape-style relighting

## Important renderer constraints

- The current Blender rendering pipeline directly supports `.glb`, `.gltf`, and `.fbx`.
- `.obj` assets should be converted to `.fbx` or `.glb` before they enter the final render manifest.
- Source objects should be staged under local folders first, then converted into a render manifest JSON with `build_external_render_manifest.py`.

## Recommended local staging layout

```text
external_sources/
  raw/
    ecommerce/
      abo/
      3d_future/
      omniobject3d/
    landscape/
      quaternius/
      kenney/
      omniobject3d/
      smithsonian/
  manifests/
```

## Phase-1 source mix

### Ecommerce target: 1000 objects

- ABO: 800
- OmniObject3D: 180
- Google Scanned Objects: 20

### Landscape target: 1000 objects

- Quaternius: 350
- Kenney: 250
- OmniObject3D: 250
- Smithsonian Open Access 3D: 150

## Download and access notes

- ABO: direct download from the official AWS open-data bucket, preferred format `.glb`
- OmniObject3D: official OpenDataLab / OpenXLab distribution, requires account login
- Google Scanned Objects: direct model access from the official Gazebo Fuel endpoint
- Quaternius: direct package download from the official website, CC0
- Kenney: direct package download from the official website, CC0
- Smithsonian Open Access 3D: select CC0 downloadable assets from the official 3D explorer

## Rendering workflow

1. Download or unpack source files into the local staging folders above.
2. Convert any `.obj` assets into `.fbx` or `.glb`.
3. Fill `external_data_plan/ecommerce_phase1_plan.json` and `external_data_plan/landscape_phase1_plan.json` root paths if needed.
4. Run `build_external_render_manifest.py` to emit renderer-ready JSON manifests under `external_sources/manifests/`.
5. Launch rendering with those manifests instead of the old Objaverse subsets.
