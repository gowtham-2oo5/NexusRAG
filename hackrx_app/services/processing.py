import asyncio
import os
import re
import subprocess
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import aiofiles
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.schema import Document


def init_executor(max_workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max_workers)


def convert_ppt_to_pdf_sync(ppt_path: str, app_logger) -> str:
    pdf_dir = os.path.dirname(ppt_path)
    try:
        result = subprocess.run(
            [r"libreoffice", "--headless", "--convert-to", "pdf", "--outdir", pdf_dir, ppt_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        base_name = os.path.splitext(os.path.basename(ppt_path))[0]
        pdf_path = os.path.join(pdf_dir, f"{base_name}.pdf")
        if os.path.exists(pdf_path):
            app_logger.info(f"✅ Converted PPT to PDF: {pdf_path}")
            return pdf_path
        raise Exception(
            f"PDF conversion failed - output file not found. LibreOffice output: {result.stdout}, Error: {result.stderr}"
        )
    except subprocess.TimeoutExpired:
        app_logger.error("LibreOffice conversion timed out after 60 seconds")
        raise Exception("PPT to PDF conversion timed out")
    except subprocess.CalledProcessError as e:
        app_logger.error(
            f"LibreOffice conversion failed: {e}, stdout: {e.stdout}, stderr: {e.stderr}"
        )
        raise Exception(f"PPT to PDF conversion failed: {e}")
    except Exception as e:
        app_logger.error(f"Unexpected error during PPT conversion: {e}")
        raise Exception(f"PPT to PDF conversion failed: {e}")


async def convert_ppt_to_pdf(ppt_path: str, executor: ThreadPoolExecutor, app_logger) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        executor, convert_ppt_to_pdf_sync, ppt_path, app_logger
    )


def extract_text_from_pdf_with_gemini_sync(pdf_path: str, app_logger) -> str:
    try:
        app_logger.info(f"Uploading {pdf_path} to Gemini...")
        pdf_file = genai.upload_file(path=pdf_path, display_name=os.path.basename(pdf_path))
        app_logger.info(f"✅ Uploaded file to Gemini: {pdf_file.name}")

        while pdf_file.state.name == "PROCESSING":
            app_logger.info("⏳ Waiting for Gemini to process file...")
            time.sleep(2)
            pdf_file = genai.get_file(pdf_file.name)

        if pdf_file.state.name == "FAILED":
            raise Exception(f"Gemini file processing failed: {pdf_file.state}")

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        prompt = (
            "Extract all the text content from this PDF document. \n"
            "Return only the text content without any additional formatting, explanations, or metadata.\n"
            "Include all text from slides, bullet points, headings, and any other textual content.\n"
            "Preserve the logical structure and flow of the content."
        )

        response = model.generate_content([prompt, pdf_file])

        try:
            genai.delete_file(pdf_file.name)
            app_logger.info(f"🗑️ Cleaned up Gemini file: {pdf_file.name}")
        except Exception as cleanup_e:
            app_logger.warning(f"Failed to cleanup Gemini file: {cleanup_e}")

        if not response.text:
            raise Exception("Gemini returned empty response")

        extracted_text = response.text.strip()
        app_logger.info(
            f"✅ Extracted {len(extracted_text)} characters from PDF using Gemini"
        )
        return extracted_text
    except Exception as e:
        app_logger.error(f"Gemini text extraction failed: {e}")
        try:
            if "pdf_file" in locals() and pdf_file:
                genai.delete_file(pdf_file.name)
                app_logger.info(f"🗑️ Cleaned up Gemini file {pdf_file.name} after error.")
        except Exception as cleanup_e:
            app_logger.error(f"Failed to clean up Gemini file after error: {cleanup_e}")
        raise Exception(f"Failed to extract text from PDF: {e}")


async def extract_text_from_pdf_with_gemini(
    pdf_path: str, executor: ThreadPoolExecutor, app_logger
) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        executor, extract_text_from_pdf_with_gemini_sync, pdf_path, app_logger
    )


def extract_text_from_image_with_gemini_sync(image_path: str, app_logger) -> str:
    try:
        app_logger.info(f"Uploading image {image_path} to Gemini...")
        image_file = genai.upload_file(
            path=image_path, display_name=os.path.basename(image_path)
        )
        app_logger.info(f"✅ Uploaded image to Gemini: {image_file.name}")

        while image_file.state.name == "PROCESSING":
            app_logger.info("⏳ Waiting for Gemini to process image...")
            time.sleep(2)
            image_file = genai.get_file(image_file.name)

        if image_file.state.name == "FAILED":
            raise Exception(f"Gemini image processing failed: {image_file.state}")

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        prompt = (
            "Extract all the text content from this image. \n"
            "This could be a document, screenshot, diagram, or any image containing text.\n"
            "Return only the text content without any additional formatting, explanations, or metadata.\n"
            "Include all visible text, labels, headings, captions, and any other textual content.\n"
            "If there are tables, preserve their structure. If there are diagrams with labels, include those labels.\n"
            "If the image contains handwritten text, do your best to transcribe it accurately."
        )

        response = model.generate_content([prompt, image_file])

        try:
            genai.delete_file(image_file.name)
            app_logger.info(f"🗑️ Cleaned up Gemini image file: {image_file.name}")
        except Exception as cleanup_e:
            app_logger.warning(f"Failed to cleanup Gemini image file: {cleanup_e}")

        if not response.text:
            raise Exception("Gemini returned empty response for image")

        extracted_text = response.text.strip()
        app_logger.info(
            f"✅ Extracted {len(extracted_text)} characters from image using Gemini"
        )
        return extracted_text
    except Exception as e:
        app_logger.error(f"Gemini image text extraction failed: {e}")
        try:
            if "image_file" in locals() and image_file:
                genai.delete_file(image_file.name)
                app_logger.info(
                    f"🗑️ Cleaned up Gemini image file {image_file.name} after error."
                )
        except Exception as cleanup_e:
            app_logger.error(f"Failed to clean up Gemini image file after error: {cleanup_e}")
        raise Exception(f"Failed to extract text from image: {e}")


async def extract_text_from_image_with_gemini(
    image_path: str, executor: ThreadPoolExecutor, app_logger
) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        executor, extract_text_from_image_with_gemini_sync, image_path, app_logger
    )


def process_excel_file_sync(excel_path: str, app_logger) -> str:
    try:
        app_logger.info(f"Processing Excel file: {excel_path}")
        excel_file = pd.ExcelFile(excel_path)
        all_sheets_text: List[str] = []
        for sheet_name in excel_file.sheet_names:
            app_logger.info(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            sheet_text = f"SHEET: {sheet_name}\n"
            sheet_text += "=" * 50 + "\n"
            if not df.empty:
                sheet_text += f"COLUMNS: {', '.join(df.columns.astype(str))}\n"
                sheet_text += f"TOTAL ROWS: {len(df)}\n"
                sheet_text += "-" * 30 + "\n"
                for index, row in df.iterrows():
                    row_text = f"ROW {index + 1}:\n"
                    for col in df.columns:
                        value = row[col]
                        if pd.isna(value):
                            value = "N/A"
                        row_text += f"  {col}: {value}\n"
                    row_text += "\n"
                    sheet_text += row_text
                sheet_text += "\n" + "SUMMARY FORMAT" + "\n"
                sheet_text += "-" * 20 + "\n"
                for index, row in df.iterrows():
                    summary_line = f"Record {index + 1}: "
                    row_parts: List[str] = []
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            row_parts.append(f"{col}={value}")
                    summary_line += ", ".join(row_parts)
                    sheet_text += summary_line + "\n"
            else:
                sheet_text += "EMPTY SHEET\n"
            sheet_text += "\n" + "=" * 50 + "\n\n"
            all_sheets_text.append(sheet_text)
        final_text = f"EXCEL FILE: {os.path.basename(excel_path)}\n"
        final_text += "TOTAL SHEETS: " + str(len(excel_file.sheet_names)) + "\n"
        final_text += "\n".join(all_sheets_text)
        app_logger.info(
            f"✅ Successfully processed Excel file with {len(excel_file.sheet_names)} sheets"
        )
        app_logger.info(f"✅ Generated {len(final_text)} characters of searchable text")
        return final_text
    except Exception as e:
        app_logger.error(f"Excel processing failed: {e}")
        raise Exception(f"Failed to process Excel file: {e}")


async def process_excel_file(excel_path: str, executor: ThreadPoolExecutor, app_logger) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        executor, process_excel_file_sync, excel_path, app_logger
    )


def process_zip_file_sync(zip_path: str, app_logger) -> str:
    try:
        app_logger.info(f"Processing ZIP file: {zip_path}")
        extracted_content: List[str] = []
        extracted_content.append(f"ZIP FILE: {os.path.basename(zip_path)}")
        extracted_content.append("=" * 60)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            extracted_content.append(f"TOTAL FILES IN ZIP: {len(file_list)}")
            extracted_content.append("-" * 40)
            extracted_content.append("FILE STRUCTURE:")
            for file_name in file_list:
                file_info = zip_ref.getinfo(file_name)
                size_mb = file_info.file_size / (1024 * 1024)
                extracted_content.append(f"  - {file_name} (Size: {size_mb:.2f} MB)")
            extracted_content.append("\n" + "-" * 40)
            extracted_content.append("EXTRACTABLE TEXT CONTENT:")
            extracted_content.append("-" * 40)
            supported_text_extensions = {
                ".txt",
                ".md",
                ".csv",
                ".json",
                ".xml",
                ".html",
                ".py",
                ".js",
                ".css",
            }
            for file_name in file_list:
                if not file_name.endswith("/"):
                    file_ext = os.path.splitext(file_name.lower())[1]
                    if file_ext in supported_text_extensions:
                        try:
                            with zip_ref.open(file_name) as file:
                                content = file.read()
                                try:
                                    text_content = content.decode("utf-8")
                                    extracted_content.append(f"\nFILE: {file_name}")
                                    extracted_content.append("~" * 30)
                                    if len(text_content) > 5000:
                                        text_content = (
                                            text_content[:5000]
                                            + "\n... [Content truncated]"
                                        )
                                    extracted_content.append(text_content)
                                except UnicodeDecodeError:
                                    extracted_content.append(
                                        f"\nFILE: {file_name} - [Binary content, cannot extract text]"
                                    )
                        except Exception as e:
                            extracted_content.append(
                                f"\nFILE: {file_name} - [Error reading file: {str(e)}]"
                            )
                    else:
                        extracted_content.append(
                            f"\nFILE: {file_name} - [Unsupported file type for text extraction: {file_ext}]"
                        )
        final_text = "\n".join(extracted_content)
        app_logger.info(
            f"✅ Successfully processed ZIP file with {len(file_list)} files"
        )
        app_logger.info(
            f"✅ Generated {len(final_text)} characters of searchable text from ZIP"
        )
        return final_text
    except Exception as e:
        app_logger.error(f"ZIP processing failed: {e}")
        raise Exception(f"Failed to process ZIP file: {e}")


async def process_zip_file(zip_path: str, executor: ThreadPoolExecutor, app_logger) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        executor, process_zip_file_sync, zip_path, app_logger
    )


def process_bin_file_sync(bin_path: str, app_logger) -> str:
    try:
        app_logger.info(f"Processing BIN file: {bin_path}")
        file_size = os.path.getsize(bin_path)
        file_size_mb = file_size / (1024 * 1024)
        extracted_content: List[str] = []
        extracted_content.append(f"BINARY FILE: {os.path.basename(bin_path)}")
        extracted_content.append("=" * 60)
        extracted_content.append(f"FILE SIZE: {file_size_mb:.2f} MB ({file_size} bytes)")
        extracted_content.append("-" * 40)
        with open(bin_path, "rb") as f:
            header_bytes = f.read(512)
        hex_header = header_bytes.hex()
        extracted_content.append(
            f"FILE HEADER (first 512 bytes in hex): {hex_header}"
        )
        file_signatures = {
            b"\x50\x4B\x03\x04": "ZIP Archive",
            b"\x50\x4B\x05\x06": "ZIP Archive (empty)",
            b"\x50\x4B\x07\x08": "ZIP Archive (spanned)",
            b"\x52\x61\x72\x21": "RAR Archive",
            b"\x7F\x45\x4C\x46": "ELF Executable",
            b"\x4D\x5A": "Windows Executable (PE)",
            b"\x89\x50\x4E\x47": "PNG Image",
            b"\xFF\xD8\xFF": "JPEG Image",
            b"\x47\x49\x46\x38": "GIF Image",
            b"\x25\x50\x44\x46": "PDF Document",
            b"\xD0\xCF\x11\xE0": "Microsoft Office Document",
        }
        detected_type = "Unknown Binary File"
        for signature, file_type in file_signatures.items():
            if header_bytes.startswith(signature):
                detected_type = file_type
                break
        extracted_content.append(f"DETECTED FILE TYPE: {detected_type}")
        extracted_content.append("-" * 40)
        printable_strings: List[str] = []
        current_string = ""
        for byte in header_bytes:
            if 32 <= byte <= 126:
                current_string += chr(byte)
            else:
                if len(current_string) >= 4:
                    printable_strings.append(current_string)
                current_string = ""
        if current_string and len(current_string) >= 4:
            printable_strings.append(current_string)
        if printable_strings:
            extracted_content.append("EXTRACTABLE STRINGS FROM HEADER:")
            for string in printable_strings[:20]:
                extracted_content.append(f"  - {string}")
            if len(printable_strings) > 20:
                extracted_content.append(
                    f"  ... and {len(printable_strings) - 20} more strings"
                )
        else:
            extracted_content.append("NO READABLE STRINGS FOUND IN HEADER")
        extracted_content.append(
            "\nNOTE: This is a binary file. Limited text extraction is possible."
        )
        extracted_content.append(
            "For comprehensive analysis, please convert to a supported text format."
        )
        final_text = "\n".join(extracted_content)
        app_logger.info(
            f"✅ Successfully analyzed BIN file ({file_size_mb:.2f} MB)"
        )
        app_logger.info(f"✅ Generated {len(final_text)} characters of analysis text")
        return final_text
    except Exception as e:
        app_logger.error(f"BIN file processing failed: {e}")
        raise Exception(f"Failed to process BIN file: {e}")


async def process_bin_file(bin_path: str, executor: ThreadPoolExecutor, app_logger) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        executor, process_bin_file_sync, bin_path, app_logger
    )


async def load_document(file_path: str, ext: str, executor: ThreadPoolExecutor, app_logger):
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
    )

    app_logger.info(f"📚 Loading document with extension: {ext}")

    if ext.lower() in ["html", "htm"]:
        app_logger.info(f"🌐 Processing HTML webpage: {file_path}")
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = await f.read()
            from bs4 import BeautifulSoup  # local import to keep module deps contained

            soup = BeautifulSoup(html_content, "lxml")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines()]
            chunks = [chunk for line in lines for chunk in line.split("  ")]
            cleaned_text = "\n".join([c for c in chunks if c])
            if not cleaned_text.strip():
                cleaned_text = "[No textual content extracted from the webpage]"
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": file_path,
                    "file_type": "html",
                    "extraction_method": "beautifulsoup_lxml",
                },
            )
            return [doc]
        except Exception as e:
            app_logger.error(f"HTML processing failed: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=f"Failed to process HTML webpage: {str(e)}")

    if ext.lower() == "zip":
        app_logger.info(f"📦 Processing ZIP file: {file_path}")
        try:
            extracted_text = await process_zip_file(file_path, executor, app_logger)
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "zip",
                    "extraction_method": "zip_analysis",
                },
            )
            return [doc]
        except Exception as e:
            app_logger.error(f"ZIP processing failed: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=f"Failed to process ZIP file: {str(e)}")

    if ext.lower() == "bin":
        app_logger.info(f"🔧 Processing BIN file: {file_path}")
        try:
            extracted_text = await process_bin_file(file_path, executor, app_logger)
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "binary",
                    "extraction_method": "binary_analysis",
                },
            )
            return [doc]
        except Exception as e:
            app_logger.error(f"BIN processing failed: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=f"Failed to process BIN file: {str(e)}")

    if ext.lower() in ["xlsx", "xls"]:
        app_logger.info(f"📊 Processing Excel file: {file_path}")
        try:
            extracted_text = await process_excel_file(file_path, executor, app_logger)
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "excel",
                    "extraction_method": "pandas_structured",
                },
            )
            return [doc]
        except Exception as e:
            app_logger.error(f"Excel processing failed: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=f"Failed to process Excel file: {str(e)}")

    if ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]:
        app_logger.info(f"🖼️ Processing image file: {file_path}")
        try:
            extracted_text = await extract_text_from_image_with_gemini(
                file_path, executor, app_logger
            )
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "image",
                    "extraction_method": "gemini_vision",
                },
            )
            return [doc]
        except Exception as e:
            app_logger.error(f"Image processing failed: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=f"Failed to process image file: {str(e)}")

    if ext.lower() in ["ppt", "pptx"]:
        app_logger.info(f"🎨 Processing PowerPoint file: {file_path}")
        try:
            pdf_path = await convert_ppt_to_pdf(file_path, executor, app_logger)
            extracted_text = await extract_text_from_pdf_with_gemini(
                pdf_path, executor, app_logger
            )
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "converted_from": ext,
                    "extraction_method": "gemini",
                },
            )
            try:
                os.remove(pdf_path)
                app_logger.info(f"🗑️ Cleaned up temporary PDF: {pdf_path}")
            except Exception as cleanup_e:
                app_logger.warning(f"Failed to cleanup PDF: {cleanup_e}")
            return [doc]
        except Exception as e:
            app_logger.error(f"PPT processing failed: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=f"Failed to process PowerPoint file: {str(e)}")

    # Fallback to standard loaders for pdf/txt/docx
    def load_document_sync(file_path: str, ext: str):
        loader = {
            "pdf": PyPDFLoader,
            "txt": lambda p: TextLoader(p, encoding="utf-8"),
            "docx": Docx2txtLoader,
        }.get(ext.lower())
        if not loader:
            raise ValueError(f"Unsupported file extension: .{ext}")
        return loader(file_path).load()

    return await asyncio.get_event_loop().run_in_executor(
        executor, load_document_sync, file_path, ext
    )


