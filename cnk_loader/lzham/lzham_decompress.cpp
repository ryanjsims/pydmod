#include <Python.h>

#define LZHAM_DEFINE_ZLIB_API
#include "lzham.h"

#include <cstring>
#include <sstream>
#include <iostream>
#include <iomanip>

#define BUF_SIZE (1024u*1024u)

//static lzham_static_lib lzham_lib{};

PyObject *lzham_exception(const char *text) {
    PyErr_SetString(PyExc_ValueError, text);
    return (PyObject *)0;
}

static PyObject* pylzham_get_version(PyObject* self, PyObject* args)
{
    int major = LZHAM_DLL_VERSION >> 8;
    int minor = LZHAM_DLL_VERSION & 0xFF;
    std::stringstream f;
    f << std::hex << major << "." << minor;
    return Py_BuildValue("s", f.str().c_str());
}

static PyObject* pylzham_decompress_init(PyObject* self, PyObject* args, PyObject* kw) {
    z_stream *stream = (z_stream*)calloc(1, sizeof(z_stream));
    if (!stream) {
        return lzham_exception("Could not allocate stream");
    }

    return PyLong_FromVoidPtr(stream);
}

static PyObject* pylzham_decompress(PyObject* self, PyObject* args, PyObject* kw) {
    uint window_bits = 20;
    PyObject *pyStream;
    z_stream *pStream;
    const unsigned char *compressed_data;
    Py_ssize_t compressed_data_len = 0;
    Py_ssize_t uncompressed_data_len = 0;

    static const char *kwlist[] = {"stream_ptr", 
                                   "compressed_data",
                                   "decompressed_size",
                                   0};
    if (!PyArg_ParseTupleAndKeywords(args, kw,
                                     "Os#n:lzham_decompress",
                                     (char **)kwlist,
                                     &pyStream,
                                     (char **)&compressed_data, &compressed_data_len, &uncompressed_data_len)) {
        return (PyObject *)0;
    }

    pStream = (z_stream*)PyLong_AsVoidPtr(pyStream);

    if(!pStream) {
        return lzham_exception("Corrupt decompression stream");
    }

    if (!compressed_data_len) {
        return PyBytes_FromStringAndSize((char *)0, 0);
    }

    uint8_t *output = (uint8_t*)malloc(uncompressed_data_len);
    if(!output){
        return lzham_exception("Could not allocate output buffer");
    }

    uint remaining = (uint)compressed_data_len - 4;
    uint index = 0;
    uint output_index = 0;

    uint8_t s_inbuf[BUF_SIZE];
    uint8_t s_outbuf[BUF_SIZE];

    memset(pStream, 0, sizeof(pStream));
    pStream->next_in = s_inbuf;
    pStream->avail_in = 0;
    pStream->next_out = s_outbuf;
    pStream->avail_out = BUF_SIZE;

    if (inflateInit2(pStream, window_bits)) {
        return lzham_exception("inflateInit2() failed");
    }

    while(1) {
        int status;
        if(!pStream->avail_in) {
            uint n = std::min(BUF_SIZE, remaining);
            std::memcpy(s_inbuf, compressed_data + index, n);
            pStream->next_in = s_inbuf;
            pStream->avail_in = n;
            index += n;
            remaining -= n;
        }

        status = inflate(pStream, Z_SYNC_FLUSH);

        if(status == Z_STREAM_END || !pStream->avail_out) {
            uint n = BUF_SIZE - pStream->avail_out;
            std::memcpy(output + output_index, s_outbuf, n);
            output_index += n;
            pStream->next_out = s_outbuf;
            pStream->avail_out = BUF_SIZE;
        }

        if(status == Z_STREAM_END) {
            break;
        } else if(status != Z_OK) {
            std::stringstream s{};
            s << "inflate() failed with status " << lzham_z_error(status);
            return lzham_exception(s.str().c_str());
        }
    }

    if (inflateEnd(pStream) != Z_OK) {
        return lzham_exception("inflateEnd() failed!");
    }

    return PyBytes_FromStringAndSize((const char*)output, uncompressed_data_len);
}

static PyMethodDef lzham_methods[] = {
    { "get_lzham_version", pylzham_get_version, METH_NOARGS, "Return the used lzham lib version" },
    { "lzham_decompress_init", (PyCFunction)pylzham_decompress_init, METH_NOARGS, "Initializes a lzham decompressor"},
    { "lzham_decompress", (PyCFunction)pylzham_decompress, METH_VARARGS | METH_KEYWORDS, "Decompresses a block of compressed data"},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef lzham_def = {
    PyModuleDef_HEAD_INIT,
    "pycnkdec",
    "Python Forgelight CNK decompressor",
    -1,
    lzham_methods
};

PyMODINIT_FUNC PyInit_cnkdec(void)
{
    return PyModule_Create(&lzham_def);
}