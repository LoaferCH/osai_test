from ast import parse
import torch
from torchvision import transforms
from PIL import Image
import argparse


def run(path, model_path):

    classes = {0:'baseball', 1:'boxing', 2:'badminton', 3:'hockey', 4:'tennis', 5:'swimming', 6:'gymnastics', 7:'basketball', 8:'chess', 9:'volleyball', 10:'formula1', 11:'football', 12:'table_tennis', 13:'shooting'}

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(path).convert('RGB')
    image_tensor = test_transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    net = torch.load(model_path)
    # adding softmax
    net = torch.nn.Sequential(
        net, 
        torch.nn.Softmax(1))


    net.eval()
    net.to('cuda:0')
    image_tensor = image_tensor.to('cuda:0')
    output = net(image_tensor)
    # print('output', output)
    result = torch.argmax(output).detach().cpu().numpy()
    print(classes[int(result)])



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', default=None, type=str)
    parser.add_argument('-model', default=None, type=str)

    args = parser.parse_args()

    run(args.image, args.model)