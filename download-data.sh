mkdir $(pwd)/data


wget  https://yazansh-public.s3.amazonaws.com/mip-data.zip -P $(pwd)/data/
unzip $(pwd)/data/mip-data.zip -d $(pwd)/data/

rm $(pwd)/data/mip-data.zip
