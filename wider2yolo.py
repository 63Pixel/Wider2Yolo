import os
import xml.etree.ElementTree as ET
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_yolo(bbox, img_width, img_height):
    xmin, ymin, width, height = bbox
    x_center = (xmin + width / 2) / img_width
    y_center = (ymin + height / 2) / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    return f"0 {x_center} {y_center} {width_norm} {height_norm}"

def create_voc_xml(filename, width, height, bounding_boxes, output_dir):
    annotation = ET.Element("annotation")

    # Folder and Filename
    ET.SubElement(annotation, "folder").text = os.path.basename(output_dir)
    ET.SubElement(annotation, "filename").text = os.path.basename(filename)

    # Source
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "WIDER Face"

    # Size
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    # Segmented
    ET.SubElement(annotation, "segmented").text = "0"

    # Objects (Bounding Boxes)
    for bbox in bounding_boxes:
        if len(bbox) != 4:
            logging.warning(f"Invalid bounding box: {bbox}")
            continue

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "face"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(bbox[0])
        ET.SubElement(bndbox, "ymin").text = str(bbox[1])
        ET.SubElement(bndbox, "xmax").text = str(bbox[0] + bbox[2])
        ET.SubElement(bndbox, "ymax").text = str(bbox[1] + bbox[3])

    # Save the XML file
    xml_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0] + '.xml')
    os.makedirs(os.path.dirname(xml_output_path), exist_ok=True)  # Create directories if they don't exist
    tree = ET.ElementTree(annotation)
    tree.write(xml_output_path)
    logging.info(f"XML saved to {xml_output_path}")

def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            logging.debug(f"Image size for {image_path}: {img.width}x{img.height}")
            return img.width, img.height
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}")
        return None, None

def filter_annotations(file_path, image_dir, output_annotations_file):
    logging.info("Checking and filtering annotation file for existing images...")
    with open(file_path, 'r') as file:
        lines = file.readlines()

    index = 0
    total_lines = len(lines)
    filtered_lines = []
    missing_images = []
    processed_images = 0

    while index < total_lines:
        line = lines[index].strip()
        if line.lower().endswith(('.jpg', '.jpeg')):
            current_image = line
            image_path = os.path.normpath(os.path.join(image_dir, current_image))
            if os.path.exists(image_path):
                filtered_lines.append(line + '\n')
                index += 1
                if index >= total_lines:
                    logging.warning(f"No bounding boxes found for image {current_image}.")
                    break

                # Read the number of bounding boxes
                num_boxes_line = lines[index].strip()
                try:
                    num_boxes = int(num_boxes_line)
                    filtered_lines.append(num_boxes_line + '\n')
                    index += 1
                except ValueError:
                    logging.warning(f"Expected number of bounding boxes after image {current_image}, but got: {num_boxes_line}")
                    index += 1
                    continue

                # Read the bounding boxes
                for _ in range(num_boxes):
                    if index >= total_lines:
                        logging.warning(f"Not enough bounding box lines for image {current_image}. Expected {num_boxes}, but found fewer.")
                        break
                    bbox_line = lines[index].strip()
                    filtered_lines.append(bbox_line + '\n')
                    index += 1
                processed_images += 1
            else:
                missing_images.append(current_image)
                index += 1
                if index >= total_lines:
                    break
                # Read the number of bounding boxes
                try:
                    num_boxes = int(lines[index].strip())
                    index += 1 + num_boxes
                except ValueError:
                    index += 1
        else:
            logging.warning(f"Unexpected line: {line}")
            index += 1

    # Write the filtered annotations
    with open(output_annotations_file, 'w') as outfile:
        outfile.writelines(filtered_lines)

    logging.info(f"Filtered annotation file created: {output_annotations_file}")
    logging.info(f"Processed images: {processed_images}")
    logging.info(f"Missing images: {len(missing_images)}")
    if len(missing_images) > 0:
        logging.warning(f"Examples of missing images:")
        for img in missing_images[:10]:  # Show only the first 10 missing images
            logging.warning(f"- {img}")
        if len(missing_images) > 10:
            logging.warning(f"... and {len(missing_images) - 10} more missing images.")

def process_annotations(file_path, output_yolo_dir, output_voc_dir, image_dir):
    logging.info("Starting annotation processing.")
    with open(file_path, 'r') as file:
        lines = file.readlines()
    logging.info(f"Number of lines in annotation file: {len(lines)}")

    index = 0
    total_lines = len(lines)
    yolo_file_count = 0
    voc_file_count = 0

    while index < total_lines:
        line = lines[index].strip()
        logging.debug(f"Processing line {index + 1}: {line}")
        if line.lower().endswith(('.jpg', '.jpeg')):
            current_image = line
            logging.info(f"Found image: {current_image}")
            index += 1
            if index >= total_lines:
                logging.warning(f"No bounding boxes found for image {current_image}.")
                break

            # Read the number of bounding boxes
            num_boxes_line = lines[index].strip()
            try:
                num_boxes = int(num_boxes_line)
                logging.info(f"Number of bounding boxes for {current_image}: {num_boxes}")
                index += 1
            except ValueError:
                logging.warning(f"Expected number of bounding boxes after image {current_image}, but got: {num_boxes_line}")
                index += 1
                continue

            bounding_boxes = []
            for _ in range(num_boxes):
                if index >= total_lines:
                    logging.warning(f"Not enough bounding box lines for image {current_image}. Expected {num_boxes}, but found fewer.")
                    break
                bbox_line = lines[index].strip()
                bbox_values = bbox_line.split()
                if len(bbox_values) >= 4:
                    try:
                        bbox = list(map(int, bbox_values[:4]))
                        bounding_boxes.append(bbox)
                        logging.debug(f"Found bounding box: {bbox}")
                    except ValueError:
                        logging.warning(f"Invalid bounding box values: {bbox_line} for image {current_image}")
                else:
                    logging.warning(f"Incomplete bounding box data: {bbox_line} for image {current_image}")
                index += 1

            # Process the current image with bounding boxes
            normalized_image_path = os.path.normpath(current_image)
            image_path = os.path.join(image_dir, normalized_image_path)
            logging.info(f"Trying to open image: {image_path}")
            img_width, img_height = get_image_size(image_path)
            if img_width is None or img_height is None:
                logging.warning(f"Skipping image due to missing size information: {current_image}")
                continue

            # Save YOLO file
            yolo_output_path = os.path.join(output_yolo_dir, os.path.splitext(normalized_image_path)[0] + '.txt')
            try:
                os.makedirs(os.path.dirname(yolo_output_path), exist_ok=True)  # Create directories if they don't exist
                with open(yolo_output_path, 'w') as yolo_file:
                    for bbox in bounding_boxes:
                        yolo_line = convert_to_yolo(bbox, img_width, img_height)
                        if yolo_line:
                            yolo_file.write(yolo_line + '\n')
                yolo_file_count += 1
                logging.info(f"YOLO saved to {yolo_output_path}")
            except Exception as e:
                logging.error(f"Error saving YOLO file: {yolo_output_path}, Error: {e}")

            # Save VOC XML file
            try:
                create_voc_xml(normalized_image_path, img_width, img_height, bounding_boxes, output_voc_dir)
                voc_file_count += 1
            except Exception as e:
                logging.error(f"Error creating VOC XML file for {current_image}: {e}")

        else:
            logging.warning(f"Unexpected line: {line}")
            index += 1

    # Summary of generated files
    logging.info(f"YOLO files created: {yolo_file_count}, VOC XML files created: {voc_file_count}")

# Example directories and parameters
annotations_file = r'YOUR PATH HERE'  # Use the validation annotation file
output_yolo_dir = r'output/yolo'
output_voc_dir = r'output/voc'
image_dir = r'YOUR PATH HERE'  # Path to validation images

# Create directories if they don't exist
os.makedirs(output_yolo_dir, exist_ok=True)
os.makedirs(output_voc_dir, exist_ok=True)

# Optional: Create a filtered annotation file
filtered_annotations_file = r'YOUR PATH HERE'
filter_annotations(annotations_file, image_dir, filtered_annotations_file)

# Process annotations with the filtered annotation file
process_annotations(filtered_annotations_file, output_yolo_dir, output_voc_dir, image_dir)
