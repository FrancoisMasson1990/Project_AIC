# 2D U-Net for Medical AIC Project Dataset

WIP
## "Step 1 of 4: Convert raw data to HDF5 file"

Run Python script to convert to a single HDF5 file
Resize should be a multiple of 16 because of the way the
max pooling and upsampling works in U-Net. The rule is
2^n where n is the number of max pooling/upsampling concatenations.
```
python convert_raw_to_hdf5.py 
```

## "Step 2 of 4: Train U-Net on dataset"
```
python train.py \
```

## "Step 3 of 4: Run sample inference script"
```
python plot_inference_examples.py  \
         --data_path $DECATHLON_DIR \
         --data_filename $MODEL_OUTPUT_FILENAME \
         --output_path $MODEL_OUTPUT_DIR \
         --inference_filename $INFERENCE_FILENAME \
         --crop_dim $IMG_SIZE
```