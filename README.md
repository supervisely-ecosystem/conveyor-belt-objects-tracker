<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/conveyor belt objects tracker/releases/download/v0.0.1/conveyor-belt-poster-new.png"/>

# Track and interpolate objects at the conveyor belt

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/conveyor belt objects tracker)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/conveyor belt objects tracker)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/conveyor belt objects tracker.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/conveyor belt objects tracker.png)](https://supervise.ly)

</div>

## Overview

**Conveyor Belt Interpolator** is an application designed to assist annotators in labeling objects by making predictions based on the conveyor belt's speed. The app supports both **images** and **videos** projects. The app offers both auto-tracking and manual tracking modes for flexibility in the annotation process. Additionally, _for images project_, each labeled object is automatically assigned a `track_id` tag.

**Perspective-distortion-free scenes only**

The application employs algorithms specifically designed for scenes where there is no perspective distortion. This makes it ideal for processing images or video captured by a static camera in environments with strictly linear motion (e.g. conveyor belts), ensuring stable visual parameters without affecting the scene's geometry.

**Low FPS data only**

The interpolation algorithm is fine-tuned for low frame rate data (e.g. 10-15 fps) where smoothness needs to be improved without introducing artefacts or distortions. The application uses temporal interpolation methods to generate additional frames, simulating a higher frame rate without increasing the original data volume.

For high frame rate data, we recommend using more advanced models, such as [MixFormer object tracking (CVPR2022)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mix-former/serve/serve), which offer precise motion restoration and perform better in dynamic scenes.

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mixformer/serve/serve" src="https://github.com/supervisely-ecosystem/MixFormer/assets/119248312/e74e2bd9-f915-48b1-bb97-ee808326dff5" width="500px" style='padding-bottom: 20px'/>

## LightGlue

[LightGlue](https://github.com/cvg/LightGlue) (ICCV 2023) is a lightweight neural network designed for image feature matching. It is used in conjunction with a feature descriptor, specifically, **SuperPoint**. LightGlue and SuperPoint form a powerful pipeline for various computer vision tasks such as image matching and localization. It achieves state-of-the-art performance on several benchmarks while being faster than previous methods.

Our application uses LightGlue to estimate a 2D translation vector (XY), that represents the movement of a conveyor between two frames. Once we obtain this vector, we can propagate all annotations to the next frame by applying the same translation to the labels.

## How To Run

**Step 1:** Open the Project.<br><br>

**Step 2:** In the `Apps` tab, run the `Conveyor Belt Auto Tracking` app. Click `Open`.<br><be>

**Step 3:** When the `Enable Autotracking` checkbox is selected, the object of interest (e.g., the PET bottle) will be automatically labeled on both the previous and following images with interpolated labels applied in real-time.

If the checkbox is not selected, the `Track Objects` button will become available, allowing manual interpolation of labels across the scene upon clicking.<br><br>

After finishing using the app, don't forget to stop the app session manually in the App Sessions.

## Acknowledgements

This app is based on the great work [LightGlue](https://github.com/cvg/LightGlue)
