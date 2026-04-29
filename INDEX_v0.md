A workflow/application that processes photos from multiple wildlife cameras that are install at birdfeeders, that recognizes and filters the photos containing one or more hummingbirds and reports the corresponding observation data + metadata.

The cameras snap motion-triggered photos. The cameras produce photos with additional metadata (temperatue `XX F`, camera id `CAMERA X`, date `DD MON YYY`, time `HH:MM AM/PM`) in a fixed-size margin at the bottom of the photo. 

Only two kinds of birds actually visit the monitored feeders: hummingbirds and bananaquits. So there are basically only three categories: hummingbird, bird (non-hummingbird), and no bird. There is a labeled training dataset of ~10.000 images with about 1.500 containing hummingbirds, and 3.150 containing bananaquits.

I envision an efficient workflow as follows:
    1. Cut bottom metadata margin from photo and process it separately using Image-to-Text step and outputting the structured metadata.
    2. Run a SOA pretrained object-recognition model like YOLO11 or YOLO26 (which has a bird class) on the photo to discard non-bird photos and crop photos with birds to rectangle(s) with bird.
    3. Adjust resolution and size of cropped bird image.
    4. Run a custom fine-tuned classification model on the bird image to distinguish hummingbird from non-hummingbird (i.e. bananaquit).
    5. Combine bird classification with the metadata and return structured output to be inserted into a results database.
    6. Post-process to remove double counting of the same bird visit since the motion detecting cameras always make three photos with 15s delays.

The custom fine-tuned model should be based on a pretrained image recognition model that already contains good internal representations so that it can be effectively fine-tuned on the two-class hummingbird vs. non-hummingbird classification task with the limited labeled training data. 

The training further should apply reasonable data augmentation (e.g. translation, rotation, contrast, brightness, etc.), with brightness being particularly important since there are photos with different lighting during day, dusk/dawn, and night that should each have similar classification performance to not bias the results. Optionally, additional hummingbird and non-hummingbird photos for training may be acquired from the iNaturalist database.
