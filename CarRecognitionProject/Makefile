VERSION=1.0.0

build-docker-image:
	docker build -t car-recognition-service:$(VERSION) --platform=linux/amd64 .
tag-docker-image-aws:
	docker tag car-recognition-service:$(VERSION) 319282596033.dkr.ecr.us-east-1.amazonaws.com/smart_city_traffic/car-recognition-service:$(VERSION)
push-docker-image-aws:
	docker push 319282596033.dkr.ecr.us-east-1.amazonaws.com/smart_city_traffic/car-recognition-service:$(VERSION)
package-build-docker-image-aws: build-docker-image tag-docker-image-aws push-docker-image-aws
