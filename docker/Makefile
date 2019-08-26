DOCKER_FILE=Dockerfile
SRC?=$(shell dirname `pwd`)

build:
	docker build -t svrgkeras -f $(DOCKER_FILE) .

bash:
	docker run -it -v $(SRC):/workspace svrgkeras bash
