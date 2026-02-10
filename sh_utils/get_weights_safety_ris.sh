#https://drive.google.com/file/d/1F3eoNlQsRY_D0Npp95jKbHBFFZO43JcU/view?usp=drive_link

apt update && apt install -y curl unzip

google_drive_id=1F3eoNlQsRY_D0Npp95jKbHBFFZO43JcU
save_to_folder=results/polamp_env/RIS/final_weights
file_name=polamp_env_ris_safety.zip

cd ..
# check if folder for weights exist
mkdir -p "$save_to_folder"

rm -rf $save_to_folder/*

# download zip with weights from google drive
curl "https://drive.usercontent.google.com/download?id={$google_drive_id}&confirm=xxx" -o $save_to_folder/$file_name

# unzip folder
unzip $save_to_folder/$file_name -d $save_to_folder

# cp weights in "final_wieghts" folder
cp -r $save_to_folder/polamp_env_ris_modified_done_100_clip_safety_5.0/* $save_to_folder

# delete zip, intermediate folder
rm -rf $save_to_folder/polamp_env_ris_modified_done_100_clip_safety_5.0
rm $save_to_folder/$file_name


echo final weights is saved in $save_to_folder