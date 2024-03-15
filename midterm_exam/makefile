init_kaggle:
	cp ./kaggle/kaggle.json ~/.kaggle/kaggle.json
	chmod 600 ~/.kaggle/kaggle.json

init_exp:
	@read -p "Enter your experiment name: " exp_name; \
	exp_folder=exp_$$exp_name; \
	mkdir -p ./$$exp_folder; \
	cp ./exp_word_count/word_count.ipynb ./$$exp_folder/$$exp_name.ipynb; \
