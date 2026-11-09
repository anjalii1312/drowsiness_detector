color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)