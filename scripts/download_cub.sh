mkdir -p data/CUB_200_2011
curl -L https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1 > data/CUB_200_2011/CUB_200_2011.tgz
tar -xvzf data/CUB_200_2011/CUB_200_2011.tgz -C data/CUB_200_2011
