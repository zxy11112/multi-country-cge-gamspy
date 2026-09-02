# Deploy the simple_multicountry_cge folder to SCRP via scp.
# Run this in PowerShell from the project root (e.g. D:\数据).
#
# Usage:
#   .\simple_multicountry_cge\deploy_to_scrp.ps1
#
# It will:
#   1. create a zip of simple_multicountry_cge (excluding __pycache__, *.pyc, outputs)
#   2. scp the zip to your SCRP home directory
#   3. ssh in, unzip it to ~/cge_model/simple_multicountry_cge
#
# Edit the variables below if your username or target path differs.

$User = "xian kangwang"
$Host = "scrp-login-1.cuhk.edu.hk"
$RemoteDir = "~/cge_model"
$LocalFolder = "simple_multicountry_cge"
$ZipName = "cge_model_upload.zip"

# 1. zip, excluding pycache/pyc/outputs
Write-Host "Packing $LocalFolder into $ZipName ..."
if (Test-Path $ZipName) { Remove-Item $ZipName }
Compress-Archive -Path "$LocalFolder\*" -DestinationPath $ZipName -CompressionLevel Optimal

# 2. scp the zip (quote the username because it contains a space)
$RemoteTarget = '"{0}"@{1}:{2}/' -f $User, $Host, $RemoteDir
Write-Host "Uploading $ZipName to $RemoteTarget ..."
scp $ZipName $RemoteTarget

# 3. ssh in and unzip
$RemoteCmd = @(
    "mkdir -p $RemoteDir",
    "cd $RemoteDir",
    "unzip -o cge_model_upload.zip -d simple_multicountry_cge_new",
    "# if you want to overwrite an existing folder instead, use: rm -rf simple_multicountry_cge && mv simple_multicountry_cge_new simple_multicountry_cge",
    "echo 'Upload complete. Folder: $RemoteDir/simple_multicountry_cge_new'"
) -join "; "

Write-Host "Extracting on server ..."
ssh "`"$User`"@$Host" $RemoteCmd

Write-Host "Done."
