rm model_data.zip
zip -qr model_data.zip model_data
echo "zip data folder successfully"
md5sum model_data.zip > model_data.md5
echo "calculate md5 successfully"
hadoop fs -rm chunk_segmentor/model_data.md5
hadoop fs -rm chunk_segmentor/model_data.zip
hadoop fs -put model_data.zip chunk_segmentor
hadoop fs -put model_data.md5 chunk_segmentor
echo "commit new data file to hdfs successfully"
PUTFILE_1 = model_data.md5
PUTFILE_2 = model_data.zip
ftp -v -n 192.168.8.23 << EOF
user yilei.wang ifchange0829FWGR
delete chunk_segmentor/model_data.md5
delete chunk_segmentor/model_data.zip
put model_data.md5 chunk_segmentor/model_data.md5
put model_data.zip chunk_segmentor/model_data.zip
bye
EOF
echo "commit new data file to ftp successfully"
