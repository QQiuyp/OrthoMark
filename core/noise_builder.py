"""Noise layer construction utilities."""

from models.noise import Noise


def build_noises(noise_type: str):
    """
    Build noise layers based on noise type.

    Args:
        noise_type: Type of noise ("JP", "NGMIX", etc.)

    Returns:
        Tuple of (test_noise, train_noise, test_train_noise)
    """
    noise_type = noise_type.upper()

    if noise_type == "JP":
        test_noise = Noise([
            "PCombined([JpegTest(50), JpegTest(60), JpegTest(70), JpegTest(80), JpegTest(90), Identity()])"
        ])
        train_noise = Noise([
            "Combined([KJpeg(50),KJpeg(60),KJpeg(70),Identity()])"
        ])
        test_train_noise = Noise([
            "Combined([KJpeg(50),KJpeg(60),KJpeg(70),Identity()])"
        ])

    elif noise_type == "NGMIX":
        test_noise = Noise([
            "PCombined(["
            "RC(91,91), Elastic((3,3)), "
            "AF(s=(-55,-55)), AF(s=(55,55)), "
            "AF(r=(-45,-45)), AF(r=(45,45)), "
            "Erase((0.5,0.5)), "
            "JpegTest(50), MF(7), GF(2), Dropout(0.5), SP(0.1), GN(0.04), "
            "Bright(0.2, 0.2), Bright(2, 2), "
            "Contrast(0.2, 0.2), Contrast(2, 2), "
            "Hue(-0.1, -0.1), Hue(0.1, 0.1), "
            "Saturation(0.2, 0.2), Saturation(2, 2)"
            "])"
        ])

        train_noise = Noise([
            "Combined(["
            "RC(91,91),"
            "Elastic((3,3)),Elastic((3,3)),Elastic((3,3)),"
            "KJpeg(40),KJpeg(50),KJpeg(60),"
            "Erase((0.5,0.5)),"
            "AF(s=(-55,-55)),AF(s=(55,55)),AF(s=(-55,55)),"
            "AF(r=(-45,-45)),AF(r=(45,45)),AF(r=(-45,45)),"
            "MF(7),"
            "GF(2),"
            "Dropout(0.5),"
            "SP(0.1),"
            "GN(0.04),"
            "Bright(0.2,0.2),Bright(2,2),"
            "Contrast(0.2,0.2),Contrast(2,2),"
            "Hue(-0.1,-0.1),Hue(0.1,0.1),"
            "Saturation(0.2,0.2),Saturation(2,2)"
            "])"
        ])

        test_train_noise = Noise([
            "Combined(["
            "RC(91,91),RC(91,91),RC(91,91),RC(91,91),RC(91,91),RC(91,91),"
            "Elastic((3,3)),Elastic((3,3)),Elastic((3,3)),Elastic((3,3)),Elastic((3,3)),Elastic((3,3)),"
            "KJpeg(50),KJpeg(50),KJpeg(50),KJpeg(40),KJpeg(40),KJpeg(40),KJpeg(60),KJpeg(60),KJpeg(60),"
            "Erase((0.5,0.5)),Erase((0.5,0.5)),Erase((0.5,0.5)),"
            "AF(s=(-55,-55)),AF(s=(55,55)),AF(s=(-55,55)),"
            "AF(r=(-45,-45)),AF(r=(45,45)),AF(r=(-45,45)),"
            "MF(7),GF(2),GF(2),Dropout(0.5),"
            "SP(0.1),SP(0.1),SP(0.1),"
            "GN(0.04),GN(0.04),GN(0.04),GN(0.04),GN(0.04),GN(0.04),"
            "Bright(0.2,0.2),Bright(2,2),Bright(2,2),"
            "Contrast(0.2,0.2),Contrast(2,2),"
            "Hue(-0.1,-0.1),Hue(0.1,0.1),"
            "Saturation(0.2,0.2),Saturation(2,2)"
            "])"
        ])

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return test_noise, train_noise, test_train_noise


def build_test_noises(noise_type: str):
    """
    Build noise layers for testing only.

    Args:
        noise_type: Type of noise

    Returns:
        Tuple of (test_noise, test_train_noise)
    """
    test_noise, _, test_train_noise = build_noises(noise_type)
    return test_noise, test_train_noise
