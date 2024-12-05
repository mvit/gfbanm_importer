# WIP GFBANM Blender Importer

This is Blender 2.80+ addon for importing single or multiple .gfbanm files from Pokémon Sword/Shield video games.
## Dependencies:
- [Flatbuffers library](https://pypi.org/project/flatbuffers/) (the addon will attempt installing it using pip if not detected)
## Seems to be working:
- Skeleton:
  - Translation transforms
  - Scaling transforms
## Currently working incorrectly:
- Skeleton:
  - Rotation transforms (affect Translation transforms)
## Not implemented:
- Material flags
- Event data

Flatbuffers schema scripts were generated from [pkZukan's gfbanm.fbs](https://github.com/pkZukan/PokeDocs/blob/main/SWSH/Flatbuffers/Animation/gfbanm.fbs).