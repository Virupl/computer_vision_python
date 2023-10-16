import cv2
import mediapipe as mp
import pyautogui


def move_cursor(x, y, width, height):
    wScr, hScr = pyautogui.size()
    # Calculate the new position of the cursor based on the webcam frame dimensions
    screen_width, screen_height = pyautogui.size()
    new_x = int(x * screen_width / width)
    new_y = int(y * screen_height / height)

    # Move the mouse cursor to the new position
    pyautogui.moveTo(new_x, new_y, duration=0.01)


def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB and process it with Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, lm in enumerate(hand_landmarks.landmark):
                    # Get the position of the index finger (LM4)
                    if idx == 4:
                        height, width, _ = frame.shape
                        x, y = int(lm.x * width), int(lm.y * height)
                        move_cursor(x, y, width, height)

                # Draw the hand landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Cursor Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
