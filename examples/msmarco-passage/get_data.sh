wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz

cd marco

join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv

TOKENIZER=bert-base-uncased
TOKENIZER_ID=bert

python ../tokenize_queries.py --tokenizer_name $TOKENIZER --query_file dev.query.txt --save_to $TOKENIZER_ID/query/dev.query.json
python ../tokenize_passages.py --tokenizer_name $TOKENIZER --file corpus.tsv --save_to $TOKENIZER_ID/corpus

cd -