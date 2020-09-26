import os
import cv2
import click
import numpy as np
from happybot.blob_detector import BlobFinder
from happybot.face_helper import faceUtil
from happybot.facs_helper import facialActions
from happybot.tf_helper import ImportGraph, facs2emotion
from happybot.constants import dict_upper, dict_lower, dict_emotion
from happybot.parameters import Parameters


def draw_infos(big_frame, params, idxFacsLow, idxFacsUp, emotion):
    """
    This function is responsible to draw the texts on result image

    :param big_frame:
    :param params:
    :param idxFacsLow:
    :param idxFacsUp:
    :param emotion:
    :return:
    """
    for idxJ, dd in enumerate(dict_upper):
        cv2.putText(big_frame, dd, (10, params.pos_upper[idxJ]), params.font, params.font_size, (255, 255, 255), 2,
                    cv2.LINE_AA)
    for idxJ, dd in enumerate(dict_lower):
        cv2.putText(big_frame, dd, (10, params.pos_lower[idxJ]), params.font, params.font_size, (255, 255, 255), 2,
                    cv2.LINE_AA)
    for idxJ, dd in enumerate(dict_emotion):
        cv2.putText(big_frame, dd, (380, params.pos_emotion[idxJ]), params.font, params.font_size, (255, 255, 255),
                    2, cv2.LINE_AA)

    # Write text on frame.
    if len(idxFacsLow) > 0:
        for ii in idxFacsLow[0]:
            cv2.putText(big_frame, dict_lower[ii], (10, params.pos_lower[ii]), params.font, params.font_size,
                        (255, 0, 0), 2,
                        cv2.LINE_AA)
    if len(idxFacsUp) > 0:
        for jj in idxFacsUp[0]:
            cv2.putText(big_frame, dict_upper[jj], (10, params.pos_upper[jj]), params.font, params.font_size,
                        (255, 0, 0), 2,
                        cv2.LINE_AA)

    cv2.putText(big_frame, dict_emotion[emotion], (380, params.pos_emotion[emotion]), params.font, params.font_size,
                (255, 0, 0), 2,
                cv2.LINE_AA)

    return big_frame


def run_detection_model(frame, params: Parameters, face_op, modelLow, modelUp):
    """
    This function apply the model in the frama passed has parameter
    :param frame:
    :param params:
    :param face_op:
    :param modelLow:
    :param modelUp:
    :return:
    """
    small_frame = cv2.resize(frame, (0, 0), fx=params.scaleFactor, fy=params.scaleFactor)

    # Get facial landmarks and position of face on image.
    vec, point = face_op.get_vec(small_frame, params.centerFixed)

    # Get facial features.
    feat = facialActions(vec, small_frame)
    newFeaturesUpper = feat.detectFeatures()
    newFeaturesLower = feat.detectLowerFeatures()

    try:
        if np.any(point):
            # Increase size of frame for viewing.
            big_frame = cv2.resize(small_frame, (0, 0), fx=params.scaleUp * 1 / params.scaleFactor,
                                   fy=params.scaleUp * 1 / params.scaleFactor)

            # Check if the face is looking directly at the camera.
            jawBool, eyeBool = feat.checkProfile(params.tol)
            if jawBool and eyeBool:
                neutralFeaturesUpper = newFeaturesUpper.copy()
                neutralFeaturesLower = newFeaturesLower.copy()

            facialMotionUp = np.reshape(feat.UpperFaceFeatures(neutralFeaturesUpper, newFeaturesUpper), (-1, 19))
            facialMotionLow = np.reshape(feat.LowerFaceFeatures(neutralFeaturesLower, newFeaturesLower), (-1, 6))

            # Predict AUs with TF model.
            facsLow = modelLow.run(facialMotionLow)
            facsUp = modelUp.run(facialMotionUp)

            # Predict emotion based on AUs.
            feel = facs2emotion(facsUp[0, :], facsLow[0, :])
            emotion = feel.declare()
            print("Emotion: ", emotion)

            # Get index of AUs.
            idxFacsLow = np.where(facsLow[0, :] == 1)
            idxFacsUp = np.where(facsUp[0, :] == 1)

            return big_frame, idxFacsLow, idxFacsUp, emotion
    except Exception as e:
        print(f"Detect fail: {e}")

    return frame, None, None, None


@click.command()
@click.option("--image", help="Image path")
@click.option("--video", help="Video path")
def main(image, video):
    """
    This main function is reponsible to start the detection process for a specifc image or video
    if empty we'll set the detection for default webam
    :param image:
    :param video:
    :return:
    """


    """
    Starting Parameters and helpers
    """
    params = Parameters()
    blob = BlobFinder(params.xCenter, params.yCenter, params.thresh, params.motor_thresh)
    face_op = faceUtil()
    dirname = os.path.dirname(__file__)
    load_file_low = os.path.join(dirname, './data/models/bottom/model-3000')
    load_file_up = os.path.join(dirname, './data/models/top/model-3000')
    modelLow = ImportGraph(load_file_low)
    modelUp = ImportGraph(load_file_up)


    """
    Core of function
    """
    if image:
        frame = cv2.imread(image)
        big_frame, idxFacsLow, idxFacsUp, emotion = run_detection_model(frame, params, face_op, modelLow, modelUp)
        big_frame = draw_infos(big_frame, params, idxFacsLow, idxFacsUp, emotion)
        cv2.imshow('frame', big_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        if video == None:
            video = 0

        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()

        while ret:
            ret, frame = cap.read()
            big_frame, idxFacsLow, idxFacsUp, emotion = run_detection_model(frame, params, face_op, modelLow, modelUp)

            if emotion != None:
                big_frame = draw_infos(big_frame, params, idxFacsLow, idxFacsUp, emotion)

            cv2.imshow('frame', big_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()