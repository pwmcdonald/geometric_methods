import enums
from os import listdir, rename, remove, mkdir
from shutil import move


def clean_data(
    raw_folder_path: str,
    container_path: str,
) -> None:
    """Renames .wav files and formats them consistently. This function is built
    specifically to parse the file structure of the Romance language data
    provided for this project.

    Args:
        raw_folder_path (str): The file path for the folder to be reformatted.

        container_path (str): The file path for the directory that will contain the
        cleaned pipeline data.

    Returns:
        None.
    """

    # French
    french_speakers = [l.value for l in enums.FrenchSpeakers]
    for speaker in french_speakers:
        folder = f"{raw_folder_path}/{enums.RomanceLanguages.FRENCH.value}{speaker}"
        if speaker == enums.FrenchSpeakers.SPEAKER_1.value:
            for _, filename in enumerate(listdir(folder)):
                if (
                    filename == "un.wav"
                    or filename == "deux.wav"
                    or filename == "trois.wav"
                    or filename == "quatre.wav"
                    or filename == "cinq.wav"
                    or filename == "six.wav"
                    or filename == "sept.wav"
                    or filename == "huit.wav"
                    or filename == "neuf.wav"
                    or filename == "dix.wav"
                ):
                    dst = f"{enums.RomanceLanguages.FRENCH.value}{speaker}_{filename}"
                    src = f"{folder}/{filename}"
                    dst = f"{folder}/{dst}"
                    rename(src, dst)

                else:
                    remove(f"{folder}/{filename}")
        else:
            for _, filename in enumerate(listdir(folder)):
                if (
                    filename == f"un{str(speaker)}.wav"
                    or filename == f"deux{speaker}.wav"
                    or filename == f"trois{speaker}.wav"
                    or filename == f"quatre{speaker}.wav"
                    or filename == f"cinq{speaker}.wav"
                    or filename == f"six{speaker}.wav"
                    or filename == f"sept{speaker}.wav"
                    or filename == f"huit{speaker}.wav"
                    or filename == f"neuf{speaker}.wav"
                    or filename == f"dix{speaker}.wav"
                ):
                    dst = f"{enums.RomanceLanguages.FRENCH.value}{speaker}_{filename[:-6]}.wav"
                    src = f"{folder}/{filename}"
                    dst = f"{folder}/{dst}"
                    rename(src, dst)

                else:
                    remove(f"{folder}/{filename}")

    # Italian
    italian_speakers = [l.value for l in enums.ItalianSpeakers]
    for speaker in italian_speakers:
        folder = f"{raw_folder_path}/{enums.RomanceLanguages.ITALIAN.value}{speaker}"

        for _, filename in enumerate(listdir(folder)):
            if (
                filename == "uno.wav"
                or filename == "due.wav"
                or filename == "tre.wav"
                or filename == "quattro.wav"
                or filename == "cinque.wav"
                or filename == "sei.wav"
                or filename == "sette.wav"
                or filename == "otto.wav"
                or filename == "nove.wav"
                or filename == "dieci.wav"
            ):
                dst = f"{enums.RomanceLanguages.ITALIAN.value}{speaker}_{filename}"
                src = f"{folder}/{filename}"
                dst = f"{folder}/{dst}"
                rename(src, dst)

            else:
                remove(f"{folder}/{filename}")

    # Brazilian Portuguese
    bp_speakers = [l.value for l in enums._BrazilianPortugueseSpeakers]
    for speaker in bp_speakers:
        folder = f"{raw_folder_path}/{enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value}{speaker}"
        if speaker == enums._BrazilianPortugueseSpeakers.SPEAKER_1.value:
            for _, filename in enumerate(listdir(folder)):
                if (
                    filename == "um.WAV"
                    or filename == "DOIS.WAV"
                    or filename == "TRES.WAV"
                    or filename == "QUATRO.WAV"
                    or filename == "CINCO.WAV"
                    or filename == "SEIS.WAV"
                    or filename == "SETE.WAV"
                    or filename == "OITO.WAV"
                    or filename == "NOVE.WAV"
                    or filename == "DEZ.WAV"
                ):
                    dst = f"{enums.RomanceLanguages.PORTUGUESE.value}{speaker}_{filename.lower()}"
                    src = f"{folder}/{filename}"
                    dst = f"{folder}/{dst}"
                    rename(src, dst)

                else:
                    remove(f"{folder}/{filename}")

        elif speaker == enums._BrazilianPortugueseSpeakers.SPEAKER_4.value:
            for _, filename in enumerate(listdir(folder)):
                if (
                    filename == "um.wav"
                    or filename == "2_DOIS.wav"
                    or filename == "3_TRES.wav"
                    or filename == "4_QUATRO.wav"
                    or filename == "5_CINCO.wav"
                    or filename == "6_SEIS.wav"
                    or filename == "7_SETE.wav"
                    or filename == "8_OITO.wav"
                    or filename == "9_NOVE.wav"
                    or filename == "10_DEZ.wav"
                ):
                    if filename == "10_DEZ.wav":
                        dst = f"{enums.RomanceLanguages.PORTUGUESE.value}{speaker}_dez.wav"
                        src = f"{folder}/{filename}"
                        dst = f"{folder}/{dst}"
                        rename(src, dst)

                    else:
                        dst = f"{enums.RomanceLanguages.PORTUGUESE.value}{speaker}_{filename[2:].lower()}"
                        src = f"{folder}/{filename}"
                        dst = f"{folder}/{dst}"
                        rename(src, dst)

                else:
                    remove(f"{folder}/{filename}")
        else:
            for _, filename in enumerate(listdir(folder)):
                if (
                    filename == "um.wav"
                    or filename == "dois.wav"
                    or filename == "tres.wav"
                    or filename == "quatro.wav"
                    or filename == "cinco.wav"
                    or filename == "seis.wav"
                    or filename == "sete.wav"
                    or filename == "oito.wav"
                    or filename == "nove.wav"
                    or filename == "dez.wav"
                ):
                    dst = f"{enums.RomanceLanguages.PORTUGUESE.value}{speaker}_{filename.lower()}"
                    src = f"{folder}/{filename}"
                    dst = f"{folder}/{dst}"
                    rename(src, dst)

                else:
                    remove(f"{folder}/{filename}")

    # Lusitanian Portuguese
    tp_speakers = [l.value for l in enums._LusitanianPortugueseSpeakers]
    for speaker in tp_speakers:
        folder = f"{raw_folder_path}/{enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value}{speaker}"
        for _, filename in enumerate(listdir(folder)):
            if (
                filename == "um.wav"
                or filename == "dois.wav"
                or filename == "tres.wav"
                or filename == "quatro.wav"
                or filename == "cinco.wav"
                or filename == "seis.wav"
                or filename == "sete.wav"
                or filename == "oito.wav"
                or filename == "nove.wav"
                or filename == "dez.wav"
            ):
                dst = f"{enums.RomanceLanguages.PORTUGUESE.value}{speaker}_{filename.lower()}"
                src = f"{folder}/{filename}"
                dst = f"{folder}/{dst}"
                rename(src, dst)

            else:
                remove(f"{folder}/{filename}")

    # American Spanish
    as_speakers = [l.value for l in enums.AmericanSpanishSpeakers]
    for speaker in as_speakers:
        folder = f"{raw_folder_path}/{enums.RomanceLanguages.AMERICAN_SPANISH.value}{speaker}"
        for _, filename in enumerate(listdir(folder)):
            if (
                filename == "uno.wav"
                or filename == "dos.wav"
                or filename == "tres.wav"
                or filename == "cuatro.wav"
                or filename == "cinco.wav"
                or filename == "seis.wav"
                or filename == "siete.wav"
                or filename == "ocho.wav"
                or filename == "nueve.wav"
                or filename == "diez.wav"
            ):
                dst = f"{enums.RomanceLanguages.AMERICAN_SPANISH.value}{speaker}_{filename.lower()}"
                src = f"{folder}/{filename}"
                dst = f"{folder}/{dst}"
                rename(src, dst)

            else:
                remove(f"{folder}/{filename}")

    # Iberian Spanish
    is_speakers = [l.value for l in enums.IberianSpanishSpeakers]
    for speaker in is_speakers:
        folder = (
            f"{raw_folder_path}/{enums.RomanceLanguages.IBERIAN_SPANISH.value}{speaker}"
        )
        if (
            speaker == enums.IberianSpanishSpeakers.SPEAKER_1.value
            or speaker == enums.IberianSpanishSpeakers.SPEAKER_7.value
        ):
            for _, filename in enumerate(listdir(folder)):
                if (
                    filename == "UNO.WAV"
                    or filename == "DOS.WAV"
                    or filename == "TRES.WAV"
                    or filename == "CUATRO.WAV"
                    or filename == "CINCO.WAV"
                    or filename == "SEIS.WAV"
                    or filename == "SIETE.WAV"
                    or filename == "OCHO.WAV"
                    or filename == "NUEVE.WAV"
                    or filename == "DIEZ.WAV"
                ):
                    dst = f"{enums.RomanceLanguages.IBERIAN_SPANISH.value}{speaker}_{filename.lower()}"
                    src = f"{folder}/{filename}"
                    dst = f"{folder}/{dst}"
                    rename(src, dst)

                else:
                    remove(f"{folder}/{filename}")

        else:
            for _, filename in enumerate(listdir(folder)):
                if (
                    filename == "uno.wav"
                    or filename == "dos.wav"
                    or filename == "tres.wav"
                    or filename == "cuatro.wav"
                    or filename == "cinco.wav"
                    or filename == "seis.wav"
                    or filename == "siete.wav"
                    or filename == "ocho.wav"
                    or filename == "nueve.wav"
                    or filename == "diez.wav"
                ):
                    dst = f"{enums.RomanceLanguages.IBERIAN_SPANISH.value}{speaker}_{filename.lower()}"
                    src = f"{folder}/{filename}"
                    dst = f"{folder}/{dst}"
                    rename(src, dst)

                else:
                    remove(f"{folder}/{filename}")

    # Make pipeline_input directory
    mkdir(f"{container_path}/cleaned_pipeline_input")

    # Make folder corresponding to each digit 1-10
    for i in range(1, 11):
        mkdir(f"{container_path}/cleaned_pipeline_input/{i}_words")

    for lang in [l.value for l in enums.RomanceLanguages]:
        if lang == enums.RomanceLanguages.FRENCH.value:
            for spkr in [s.value for s in enums.FrenchSpeakers]:
                for fl in listdir(f"{raw_folder_path}/{lang}{spkr}"):
                    if enums._FrenchDict.FRENCH_1.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/1_words",
                        )
                    elif enums._FrenchDict.FRENCH_2.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/2_words",
                        )
                    elif enums._FrenchDict.FRENCH_3.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/3_words",
                        )
                    elif enums._FrenchDict.FRENCH_4.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/4_words",
                        )
                    elif enums._FrenchDict.FRENCH_5.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/5_words",
                        )
                    elif enums._FrenchDict.FRENCH_6.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/6_words",
                        )
                    elif enums._FrenchDict.FRENCH_7.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/7_words",
                        )
                    elif enums._FrenchDict.FRENCH_8.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/8_words",
                        )
                    elif enums._FrenchDict.FRENCH_9.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/9_words",
                        )
                    elif enums._FrenchDict.FRENCH_10.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/10_words",
                        )
        elif lang == enums.RomanceLanguages.ITALIAN.value:
            for spkr in [s.value for s in enums.ItalianSpeakers]:
                for fl in listdir(f"{raw_folder_path}/{lang}{spkr}"):
                    if enums._ItalianDict.ITALIAN_1.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/1_words",
                        )
                    elif enums._ItalianDict.ITALIAN_2.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/2_words",
                        )
                    elif enums._ItalianDict.ITALIAN_3.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/3_words",
                        )
                    elif enums._ItalianDict.ITALIAN_4.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/4_words",
                        )
                    elif enums._ItalianDict.ITALIAN_5.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/5_words",
                        )
                    elif enums._ItalianDict.ITALIAN_6.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/6_words",
                        )
                    elif enums._ItalianDict.ITALIAN_7.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/7_words",
                        )
                    elif enums._ItalianDict.ITALIAN_8.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/8_words",
                        )
                    elif enums._ItalianDict.ITALIAN_9.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/9_words",
                        )
                    elif enums._ItalianDict.ITALIAN_10.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/10_words",
                        )
        elif lang == enums.RomanceLanguages._BRAZILIAN_PORTUGUESE.value:
            for spkr in [s.value for s in enums._BrazilianPortugueseSpeakers]:
                for fl in listdir(f"{raw_folder_path}/{lang}{spkr}"):
                    if (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_1.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/1_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_2.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/2_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_3.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/3_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_4.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/4_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_5.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/5_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_6.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/6_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_7.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/7_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_8.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/8_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_9.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/9_words",
                        )
                    elif (
                        enums._BrazilianPortugueseDict.BRAZILIAN_PORTUGUESE_10.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/10_words",
                        )
        elif lang == enums.RomanceLanguages._LUSITANIAN_PORTUGUESE.value:
            for spkr in [s.value for s in enums._LusitanianPortugueseSpeakers]:
                for fl in listdir(f"{raw_folder_path}/{lang}{spkr}"):
                    if (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_1.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/1_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_2.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/2_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_3.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/3_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_4.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/4_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_5.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/5_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_6.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/6_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_7.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/7_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_8.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/8_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_9.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/9_words",
                        )
                    elif (
                        enums._LusitanianPortugueseDict.LUSITANIAN_PORTUGUESE_10.value
                        in fl
                    ):
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/10_words",
                        )
        elif lang == enums.RomanceLanguages.AMERICAN_SPANISH.value:
            for spkr in [s.value for s in enums.AmericanSpanishSpeakers]:
                for fl in listdir(f"{raw_folder_path}/{lang}{spkr}"):
                    if enums._AmericanSpanishDict.AMERICAN_SPANISH_1.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/1_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_2.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/2_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_3.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/3_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_4.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/4_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_5.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/5_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_6.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/6_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_7.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/7_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_8.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/8_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_9.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/9_words",
                        )
                    elif enums._AmericanSpanishDict.AMERICAN_SPANISH_10.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/10_words",
                        )
        elif lang == enums.RomanceLanguages.IBERIAN_SPANISH.value:
            for spkr in [s.value for s in enums.IberianSpanishSpeakers]:
                for fl in listdir(f"{raw_folder_path}/{lang}{spkr}"):
                    if enums._IberianSpanishDict.IBERIAN_SPANISH_1.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/1_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_2.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/2_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_3.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/3_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_4.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/4_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_5.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/5_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_6.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/6_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_7.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/7_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_8.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/8_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_9.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/9_words",
                        )
                    elif enums._IberianSpanishDict.IBERIAN_SPANISH_10.value in fl:
                        move(
                            src=f"{raw_folder_path}/{lang}{spkr}/{fl}",
                            dst=f"{container_path}/cleaned_pipeline_input/10_words",
                        )
