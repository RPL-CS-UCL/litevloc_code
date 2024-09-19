## Instraction of Benchmarking Map-free Evaluation

### Installation
Install [map-free reloc](https://github.com/nianticlabs/map-free-reloc) report
```bash
git clone https://github.com/nianticlabs/map-free-reloc
```

### Download dataset
We follow [Map-free reloc](https://github.com/nianticlabs/map-free-reloc) to structure the data
```
matterport3d/
├── map_free_eval/
    ├── test/
        ├── s00000
            ├── intrinsics.txt
            ├── poses.txt
            ├── seq0
            │   ├── frame_00000.jpg
            │   ├── frame_00000.zed.png
            └── seq1
                ├── frame_00000.jpg
                ├── frame_00001.jpg
                ├── frame_00002.jpg
                ├── ...
```

### Available Models
You can choose any of the following methods (input to `get_matcher()`):

**Dense**: ```roma, tiny-roma, dust3r, mast3r```

**Semi-dense**: ```loftr, eloftr, se2loftr, aspanformer, matchformer, xfeat-star```

**Sparse**: ```[sift, superpoint, disk, aliked, dedode, doghardnet, gim, xfeat]-lg, dedode, steerers, dedode-kornia, [sift, orb, doghardnet]-nn, patch2pix, superglue, r2d2, d2net,  gim-dkm, xfeat, omniglue, [dedode, xfeat, aliked]-subpx```


### Use
Setup path for dataset and matcher for your evaluation. We support above image matchers defined in [image-matching-models](https://github.com/gmberton/image-matching-models)
```bash
bash scripts/run_benchmark_loc_submission.py
```
Perform evaluation using [Mickey Evaluation Scripts](https://github.com/nianticlabs/mickey)
```bash
bash scripts/run_benchmark_loc_evaluation.sh
```
You can output a file named ```results/report_evaluation_025_5.txt``` with content similar to
```bash
Evaluate matching methods: master_pnp
{
  "Maximum Translation Error [m]": 0.13964442928576778,
  "Maximum Rotation Error [deg]": 3.3407354525107658,
  "Average Median Translation Error [m]": 0.01410202978829861,
  "Average Median Rotation Error [deg]": 0.3581160726250012,
  "Average Median Reprojection Error [px]": 1.033202960307021,
  "Precision @ Pose Error < (25.0cm, 5deg)": 0.9473684210526315,
  "AUC @ Pose Error < (25.0cm, 5deg)": 0.9473684430122375,
  "Precision @ VCRE < 90px": 0.9473684210526315,
  "AUC @ VCRE < 90px": 0.9473684430122375,
  "Estimates for % of frames": 0.9473684210526315
}
```
