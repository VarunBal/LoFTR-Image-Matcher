# LOFTR Image Matcher

The LOFTR Image Matcher is a Flask application that uses the Kornia LOFTR method for feature matching between two input images. It provides a simple web interface to upload images, select a pretrained model, and visualize the matched features. Additionally, it provides a REST API endpoint to perform image matching programmatically.

## Getting Started

These instructions will help you set up and run the LOFTR Image Matcher using Docker Compose.

### Prerequisites

- Docker: Make sure you have Docker installed on your system. You can download Docker from the official website: [https://www.docker.com/](https://www.docker.com/)

### Usage

Clone the repository:

```
git clone https://github.com/VarunBal/LoFTR-Image-Matcher.git
```

Change to the project directory:
```
cd LoFTR-Image-Matcher
```

Start the application using Docker Compose:
```
docker-compose up
```
***Note: Depending on the internet speed it could take a significant time to build the image.***

### Access the LOFTR Image Matcher:

Open a web browser and go to http://localhost:5000. You should see the LOFTR Image Matcher interface.

Upload Images and Perform Matching:
- Click on the "Choose File" button next to "Image 1" and "Image 2" to select the input images.
- Select a pretrained model from the "Model" dropdown. The default option is "outdoor".
- Click the "Match Images" button to perform the feature matching.
- The matched features will be visualized, and the result will be displayed on a new page.

### Using the REST API with Postman

The LOFTR Image Matcher also provides a REST API endpoint to perform image matching programmatically. You can use a tool like Postman to send a POST request to http://localhost:5000/match_api with the input images and model selection as form data. The API will return the full URL to the output image.

Example API request using Postman:

- Open Postman and create a new request.
- Set the request URL to http://localhost:5000/match_api.
- Set the request method to POST.
- Select the "Body" tab and choose "form-data".
- Add the following key-value pairs:
  - Key: image1, Value: Select File (choose the first input image file).
  - Key: image2, Value: Select File (choose the second input image file).
  - Key: model, Value: outdoor (or any other model choice from 'indoor' and 'indoor_new').
- Click the "Send" button to perform the request.
- The API response will contain the full URL to the output image.