echo "Linking dependencies from MDM"

echo "Enter the full path to your previous Install MDM [ENTER]:"
read full_mdm_path


ln -s "$full_mdm_path"/body_models/smpl ./body_models/.
ln -s "$full_mdm_path"/glove .
ln -s "$full_mdm_path"/t2m .

ln -s "$full_mdm_path"/dataset/HumanML3D ./dataset/.

