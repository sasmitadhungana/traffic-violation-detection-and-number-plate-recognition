import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Upload, MapPin, Camera, CalendarDays, FileVideo, Sparkles, Film, Check, FileType, Loader2 } from "lucide-react";
import { motion } from "framer-motion";

const UploadVideo = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const navigate = useNavigate();

  const handleUpload = async () => {
    if (!selectedFile) return;
    
    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);
    
    try {
      const response = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      
      if (response.ok) {
        navigate("/video-library");
      } else {
        alert("Upload failed.");
      }
    } catch (err) {
      console.error(err);
      alert("An error occurred during upload.");
    } finally {
      setIsUploading(false);
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const supportedFormats = [
    { name: "MP4", color: "bg-blue-50 text-blue-600 border-blue-100" },
    { name: "AVI", color: "bg-amber-50 text-amber-600 border-amber-100" },
    { name: "MOV", color: "bg-purple-50 text-purple-600 border-purple-100" },
  ];

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="space-y-8 max-w-6xl mx-auto"
    >
      <motion.div 
        variants={itemVariants}
        className="flex items-center gap-4 bg-white/50 p-6 rounded-[2rem] border border-white/60 shadow-sm backdrop-blur-md"
      >
        <div className="p-4 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl shadow-lg shadow-indigo-500/20 text-white">
          <Upload size={28} />
        </div>
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Upload Video</h1>
          <p className="text-slate-500 mt-1 font-medium text-sm">Submit new traffic footage and configure analysis metadata for AI processing.</p>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
        {/* Drag & Drop Area */}
        <motion.div 
          variants={itemVariants}
          className="lg:col-span-3"
        >
          <motion.div 
            whileHover={{ scale: 1.005 }}
            className="bg-white/80 backdrop-blur-xl rounded-[2.5rem] border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-2 h-full min-h-[450px] flex flex-col group transition-all duration-300 hover:shadow-[0_12px_40px_rgb(0,0,0,0.08)] relative overflow-hidden"
          >
            {/* Subtle background glow */}
            <div className={`absolute inset-0 bg-indigo-500/10 blur-[100px] transition-opacity duration-500 ${isDragging ? 'opacity-100' : 'opacity-0'}`}></div>

            <motion.div 
              animate={isDragging ? { scale: 0.98, borderColor: "#6366f1", backgroundColor: "rgba(238, 242, 255, 0.8)" } : { scale: 1, borderColor: "#e2e8f0", backgroundColor: "rgba(248, 250, 252, 0.5)" }}
              transition={{ type: "spring", stiffness: 400, damping: 25 }}
              className="border-2 border-dashed rounded-[2rem] h-full flex flex-col items-center justify-center relative z-10 m-1 cursor-pointer overflow-hidden"
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleFileDrop}
              onClick={() => document.getElementById('file-input')?.click()}
            >
              {selectedFile ? (
                <motion.div 
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="text-center"
                >
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 300, damping: 15, delay: 0.1 }}
                    className="h-28 w-28 bg-emerald-50 rounded-3xl shadow-xl flex items-center justify-center mb-6 mx-auto border-2 border-emerald-200"
                  >
                    <Check size={48} className="text-emerald-600" strokeWidth={3} />
                  </motion.div>
                  <h2 className="text-2xl font-black text-slate-800 mb-2 tracking-tight">File Selected</h2>
                  <p className="text-slate-600 font-bold text-sm mb-1">{selectedFile.name}</p>
                  <p className="text-slate-400 text-xs font-semibold">{(selectedFile.size / (1024 * 1024)).toFixed(1)} MB</p>
                  <motion.button 
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => { e.stopPropagation(); setSelectedFile(null); }}
                    className="mt-6 px-6 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-bold text-slate-600 hover:border-rose-300 hover:text-rose-600 transition-colors"
                  >
                    Choose Different File
                  </motion.button>
                </motion.div>
              ) : (
                <>
                  <motion.div 
                    animate={isDragging ? { y: -10, scale: 1.1, rotate: 5 } : { y: 0, scale: 1, rotate: 0 }}
                    transition={{ type: "spring", stiffness: 300, damping: 15 }}
                    className={`h-28 w-28 bg-white rounded-3xl shadow-xl flex items-center justify-center mb-8 transition-colors ${isDragging ? 'text-indigo-600 shadow-indigo-500/20' : 'text-indigo-400 shadow-slate-200/50 group-hover:text-indigo-500'}`}
                  >
                    <Upload size={44} strokeWidth={2} />
                  </motion.div>
                  <h2 className="text-2xl font-black text-slate-800 mb-3 tracking-tight">Drag & Drop Video Here</h2>
                  <p className="text-slate-500 max-w-sm text-center mb-6 text-sm font-medium leading-relaxed">
                    For best AI detection results, use 1080p resolution or higher.
                  </p>
                  
                  {/* Supported formats */}
                  <div className="flex items-center gap-2 mb-8">
                    {supportedFormats.map(f => (
                      <span key={f.name} className={`px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider border shadow-sm ${f.color} flex items-center gap-1.5`}>
                        <FileType size={12} />
                        {f.name}
                      </span>
                    ))}
                    <span className="text-xs font-semibold text-slate-400 ml-1">up to 2GB</span>
                  </div>
                  
                  <motion.button 
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => { e.stopPropagation(); document.getElementById('file-input')?.click(); }}
                    className="px-10 py-4 bg-slate-900 text-white rounded-2xl font-bold shadow-lg hover:shadow-slate-900/30 transition-all relative overflow-hidden group/btn cursor-pointer"
                  >
                    <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white opacity-20 group-hover/btn:animate-shine"></div>
                    Browse Computer
                  </motion.button>
                </>
              )}
              <input id="file-input" type="file" className="hidden" accept="video/*" onChange={handleFileSelect} />
            </motion.div>
          </motion.div>
        </motion.div>

        {/* Configuration Form */}
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="lg:col-span-2 bg-white/90 backdrop-blur-xl rounded-[2.5rem] border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] p-8 lg:p-10 relative overflow-hidden"
        >
          <motion.div variants={itemVariants} className="flex items-center gap-3 mb-8">
            <div className="p-2 bg-indigo-50 text-indigo-600 rounded-xl">
              <Sparkles size={20} />
            </div>
            <h3 className="text-xl font-extrabold text-slate-800">Video Metadata</h3>
          </motion.div>
          
          <form className="space-y-6">
            <motion.div variants={itemVariants} className="space-y-2 group">
              <label className="block text-sm font-bold text-slate-700 ml-1">Video Title</label>
              <div className="relative">
                <FileVideo size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
                <input type="text" placeholder="e.g., Highway 90 Evening Feed" className="w-full pl-12 pr-4 py-3.5 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-semibold text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all hover:border-slate-300" />
              </div>
            </motion.div>
            
            <motion.div variants={itemVariants} className="space-y-2 group">
              <label className="block text-sm font-bold text-slate-700 ml-1">Location Identifier</label>
              <div className="relative">
                <MapPin size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
                <input type="text" placeholder="e.g., Intersection 4th & Main" className="w-full pl-12 pr-4 py-3.5 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-semibold text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all hover:border-slate-300" />
              </div>
            </motion.div>

            <motion.div variants={itemVariants} className="space-y-2 group">
              <label className="block text-sm font-bold text-slate-700 ml-1">Camera Source</label>
              <div className="relative">
                <Camera size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors z-10 pointer-events-none" />
                <select className="w-full pl-12 pr-10 py-3.5 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-semibold text-slate-700 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all hover:border-slate-300 appearance-none cursor-pointer">
                  <option value="" disabled selected>Select Camera...</option>
                  <option>CAM-01 (North)</option>
                  <option>CAM-02 (South)</option>
                  <option>Dashcam Array</option>
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-400"></div>
              </div>
            </motion.div>

            <motion.div variants={itemVariants} className="space-y-2 group">
              <label className="block text-sm font-bold text-slate-700 ml-1">Date Recorded</label>
              <div className="relative">
                <CalendarDays size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
                <input type="date" className="w-full pl-12 pr-4 py-3.5 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-semibold text-slate-700 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all hover:border-slate-300 cursor-pointer" />
              </div>
            </motion.div>
            
            <motion.div variants={itemVariants} className="pt-8 mt-4 border-t border-slate-100">
              <motion.button 
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="button" 
                onClick={handleUpload}
                disabled={!selectedFile || isUploading}
                className={`w-full py-4 font-bold rounded-2xl border transition-all cursor-pointer relative overflow-hidden group ${
                  selectedFile 
                    ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 text-white border-transparent shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/40' 
                    : 'bg-slate-100 text-slate-400 border-slate-200 hover:bg-slate-200'
                } disabled:opacity-70 disabled:cursor-not-allowed`}
              >
                {selectedFile && !isUploading && (
                  <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white opacity-20 group-hover:animate-shine"></div>
                )}
                <span className="relative z-10 flex items-center justify-center gap-2">
                  {!selectedFile ? (
                    'Upload a File to Proceed'
                  ) : isUploading ? (
                    <>
                      <Loader2 size={18} className="animate-spin" />
                      Uploading & Processing...
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      Start AI Analysis
                    </>
                  )}
                </span>
              </motion.button>
            </motion.div>
          </form>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default UploadVideo;