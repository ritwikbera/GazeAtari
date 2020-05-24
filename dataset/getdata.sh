#!/bin/bash

curl -s -o action_enums.txt https://zenodo.org/record/3451402/files/action_enums.txt?download=1
curl -s -o meta_data.csv https://zenodo.org/record/3451402/files/meta_data.csv?download=1

game="alien"

curl -o "${game}.zip" "https://zenodo.org/record/3451402/files/${game}.zip?download=1"
unzip -j "${game}.zip" -d ${game}

cd ${game}
current_dir="$(basename $PWD)"

echo ${current_dir}

if [[ ${current_dir} != ${game} ]]; then
    echo "Not in ${game} directory !!!"
    exit 1
fi

mkdir extracted
find ./ -type f -name '*.tar.*' | xargs -i tar -xvf {} -C ./extracted/

