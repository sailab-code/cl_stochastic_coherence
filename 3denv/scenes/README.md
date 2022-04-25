# Stochastic Coherence Over Attention Trajectory For Continuous Learning In Video Streams

In this README you will find instructions to load the scenes used in the paper into SAILenv.

## Content

This directory contains 3 .unity files and 3 associated directories.

- EmptySpace.unity: contains the scene named ```EmptySpace``` in the paper.
- LivingRoom.unity: contains the scene named ```LivingRoom``` in the paper.
- Solid.unity: contains the scene named ```Solid``` in the paper.

## How to install

1. Download the latest SAILenv source code from [SAILenv site](https://sailab.diism.unisi.it/sailenv/) and extract
   the files into a directory. (Unix users: the `downloader.sh` script downloads SAILenv source and copies the scenes)
2. Download Unity 2019.4.2f1 from [Unity website](https://www.unity.com). Using Unity HUB is the easiest way.
   **NOTE: to use Unity you will need to create an account with a personal free license (Unity Personal).**
3. Open the SAILenv directory with Unity. The first time it will take around 15 minutes to fully load.
4. Copy the content of this directory into `Assets/Scenes`
5. Through the Unity Editor, open the file `Assets/Settings/AvailableScenesSettings`.
6. The inspector will show a list of the available scenes. Change the textbox named Size from 4 to 7. This will
   create three new empty boxes that can be filled by dragging the scenes objects from the Project view to the scene box
   in the Inspector and assign names to the scenes in the textbox below (the scenes will be
   exposed through the Python API with these names - see the example scripts in the [official site](https://sailab.diism.unisi.it/sailenv/).
7. Press the `Add (buildIndex N)` and then confirm Add as enabled for each of the newly added scenes.
8. On the top bar, select `SAILenv/Builds/Local AssetBundles`. It will take a while to compile the resources.
9. Open the "Main Menu" scene in `Assets/Scenes` and press the Play button on the top. There is no need to interact with
   this scene, just proceed to the next step.
10. The Environment is now running and you can use the Python API to open the new scenes through the names you have
    given. 