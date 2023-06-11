# URL of image to use. Change this to your image.
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
IMAGE_URL = 'https://samples.clarifai.com/metro-north.jpg'


channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

metadata = (('authorization', 'Key ' + 'API'),)

userDataObject = resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')

post_model_outputs_response = stub.PostModelOutputs(
    service_pb2.PostModelOutputsRequest(
        user_app_id=userDataObject,
        model_id='general-image-recognition',
        # version_id='aa7f35c01e0642fda5cf400f543e7c40',  ## This is optional. Defaults to the latest model version.
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        url=IMAGE_URL
                    )
                )
            )
        ]
    ),
    metadata=metadata
)

if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
    print(post_model_outputs_response.status)
    raise Exception("Post model outputs failed, status: " +
                    post_model_outputs_response.status.description)

# Since we have one input, one output will exist here.
output = post_model_outputs_response.outputs[0]

print("Predicted concepts:")
for concept in output.data.concepts:
    print("%s %.2f" % (concept.name, concept.value))

# Uncomment this line to print the full Response JSON
# print(post_model_outputs_response)
