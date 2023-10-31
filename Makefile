#create env  
create-env:
	conda create --name $(ENV_NAME) python=3.7 -y

#install requirements
install-requirements:
	pip install -r requirements.txt 


