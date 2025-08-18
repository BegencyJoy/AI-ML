import cv2


    
cam = cv2.VideoCapture(0)
current_filter_mode = 0 

while cam.isOpened():
    a, frame = cam.read()
   

    processed_frame = frame.copy() 

    if current_filter_mode == 0:
        # Normal view
        processed_frame = frame
        window_title = 'Normal View (Press 1, 2, 3 for filters, S to save, Q to quit)'

    elif current_filter_mode == 1:
        # Blurred filter
        processed_frame = cv2.GaussianBlur(frame, (69, 69), 0) # Adjust kernel size for more/less blur
        window_title = 'Blurred View (Press 0, 2, 3 for filters, S to save, Q to quit)'

    elif current_filter_mode == 2:
        # Color Sketch filter
        blurred_frame_for_sketch = cv2.bilateralFilter(frame, 9, 75, 75)
        grayscale_for_edges = cv2.cvtColor(blurred_frame_for_sketch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscale_for_edges, 50, 100) # Canny outputs white edges on black background
        inverted_edges = cv2.bitwise_not(edges) # Invert for black edges on white (mask)
            
        # Combine blurred color with black edges
        processed_frame = cv2.bitwise_and(blurred_frame_for_sketch, blurred_frame_for_sketch, mask=inverted_edges)
        window_title = 'Color Sketch (Press 0, 1, 3 for filters, S to save, Q to quit)'

    elif current_filter_mode == 3:
        # Black-and-White Pencil Sketch filter
        grayscale_for_edges = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscale_for_edges, 50, 100) # Adjust thresholds for more/less detail
        processed_frame = cv2.bitwise_not(edges) # Invert to get black edges on white background
        window_title = 'Pencil Sketch (Press 0, 1, 2 for filters, S to save, Q to quit)'
        
    # Display the processed frame
    cv2.imshow(window_title, processed_frame)

        # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    # Change filter mode based on key press
    if key == ord('0'):
        current_filter_mode = 0
    elif key == ord('1'):
        current_filter_mode = 1
    elif key == ord('2'):
        current_filter_mode = 2
    elif key == ord('3'):
        current_filter_mode = 3
    elif key == ord('s'):
        # Save the currently displayed image
        save_path = 'captured_image_filter_{}.png'.format(current_filter_mode)
        cv2.imwrite(save_path, processed_frame)
        print(f"Image saved as {save_path}")
            
        # Display the captured image in a new window and wait for a key press
        cv2.imshow('Captured Image', processed_frame)
        print("Displaying captured image. Press any key to continue.")
        cv2.waitKey(0) 

        
    elif key == ord('q'):
        break

    

    
cam.release()
cv2.destroyAllWindows()


        


