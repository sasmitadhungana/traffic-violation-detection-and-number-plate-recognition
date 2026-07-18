import React, { useState } from "react";
import { Search, Filter, Video, CheckCircle2, Clock, AlertTriangle, Download, Trash2, Eye, LayoutGrid, List, Film } from "lucide-react";
import { motion } from "framer-motion";

const VideoLibrary = () => {
  const [currentPage, setCurrentPage] = useState(1);

  const [videos, setVideos] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  React.useEffect(() => {
    fetch('http://localhost:8000/api/analysis')
      .then(res => res.json())
      .then(data => {
        const mappedVideos = data.map(v => ({
          id: v.id,
          name: v.filename,
          date: new Date(v.upload_time).toLocaleString(),
          status: v.status,
          violations: v.violation_count || 0,
          size: "Unknown", 
          duration: "Unknown"
        }));
        setVideos(mappedVideos);
        setIsLoading(false);
      })
      .catch(err => {
        console.error("Error fetching videos:", err);
        setIsLoading(false);
      });
  }, []);

  const statusCounts = {
    total: videos.length,
    completed: videos.filter(v => v.status === "Completed").length,
    processing: videos.filter(v => v.status === "Processing").length,
    failed: videos.filter(v => v.status === "Failed").length,
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.08 }
    }
  };

  const rowVariants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  const totalPages = 6;

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div 
        variants={rowVariants}
        className="flex flex-col md:flex-row md:items-end justify-between bg-white/50 p-6 rounded-[2rem] border border-white/60 shadow-sm backdrop-blur-md"
      >
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Video Library</h1>
          <p className="text-slate-500 font-medium text-sm mt-1.5">Manage and review uploaded traffic footage</p>
        </div>
        <div className="flex gap-3 mt-4 md:mt-0">
          <div className="relative group">
            <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
            <input type="text" placeholder="Search videos..." className="pl-12 pr-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 shadow-sm transition-all w-64" />
          </div>
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="flex items-center gap-2 px-5 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-bold text-slate-700 hover:bg-slate-50 hover:text-indigo-600 hover:border-indigo-200 shadow-sm transition-all cursor-pointer"
          >
            <Filter size={18} /> Filter
          </motion.button>
        </div>
      </motion.div>

      {/* Stats Bar */}
      <motion.div variants={rowVariants} className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatPill label="Total Videos" value={statusCounts.total} icon={Film} color="text-slate-600" bg="bg-slate-50 border-slate-100" />
        <StatPill label="Completed" value={statusCounts.completed} icon={CheckCircle2} color="text-emerald-600" bg="bg-emerald-50 border-emerald-100" />
        <StatPill label="Processing" value={statusCounts.processing} icon={Clock} color="text-blue-600" bg="bg-blue-50 border-blue-100" />
        <StatPill label="Failed" value={statusCounts.failed} icon={AlertTriangle} color="text-rose-600" bg="bg-rose-50 border-rose-100" />
      </motion.div>

      {/* Data Table */}
      <motion.div 
        variants={rowVariants}
        className="bg-white/90 backdrop-blur-xl rounded-[2rem] border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm whitespace-nowrap">
            <thead className="bg-slate-50/50 border-b border-slate-100 text-xs uppercase tracking-widest text-slate-500 font-bold">
              <tr>
                <th className="px-6 py-5">Video File</th>
                <th className="px-6 py-5">Upload Date</th>
                <th className="px-6 py-5">Duration &bull; Size</th>
                <th className="px-6 py-5">Status</th>
                <th className="px-6 py-5">Violations</th>
                <th className="px-6 py-5 text-right">Actions</th>
              </tr>
            </thead>
            <motion.tbody 
              variants={containerVariants}
              initial="hidden"
              animate="show"
              className="divide-y divide-slate-100/50"
            >
              {videos.map((vid) => (
                <motion.tr 
                  variants={rowVariants}
                  key={vid.id} 
                  className="hover:bg-indigo-50/30 transition-all duration-200 group relative"
                >
                  {/* Left accent bar on hover */}
                  <td className="px-6 py-5 flex items-center gap-4 relative">
                    <div className="absolute left-0 top-2 bottom-2 w-1 bg-indigo-500 rounded-r-full opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
                    <div className="p-3 bg-indigo-50 text-indigo-600 rounded-xl group-hover:bg-indigo-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-indigo-500/30 transition-all duration-300">
                      <Video size={20} />
                    </div>
                    <span className="font-bold text-slate-800 group-hover:text-indigo-600 transition-colors cursor-pointer">{vid.name}</span>
                  </td>
                  <td className="px-6 py-5 text-slate-500 font-semibold">{vid.date}</td>
                  <td className="px-6 py-5 text-slate-500 font-semibold">
                    <span className="text-slate-800">{vid.duration}</span> &bull; {vid.size}
                  </td>
                  <td className="px-6 py-5">
                    {vid.status === "Completed" && 
                      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-100 text-emerald-700 text-[11px] font-bold uppercase tracking-wider shadow-sm">
                        <CheckCircle2 size={14} className="text-emerald-500" /> Completed
                      </span>
                    }
                    {vid.status === "Processing" && 
                      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-50 border border-blue-100 text-blue-700 text-[11px] font-bold uppercase tracking-wider shadow-sm">
                        <Clock size={14} className="text-blue-500 animate-spin-slow" /> Processing
                      </span>
                    }
                    {vid.status === "Failed" && 
                      <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-50 border border-rose-100 text-rose-700 text-[11px] font-bold uppercase tracking-wider shadow-sm">
                        <AlertTriangle size={14} className="text-rose-500" /> Failed
                      </span>
                    }
                  </td>
                  <td className="px-6 py-5">
                    {vid.violations > 0 ? 
                      <span className="inline-flex px-3 py-1.5 rounded-lg bg-orange-50 border border-orange-100 text-orange-700 text-[11px] font-bold uppercase tracking-wider shadow-sm">
                        {vid.violations} Found
                      </span> : 
                      <span className="inline-flex px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-100 text-slate-400 text-[11px] font-bold uppercase tracking-wider">
                        None
                      </span>
                    }
                  </td>
                  <td className="px-6 py-5 text-right">
                    <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <motion.button whileHover={{ scale: 1.1 }} className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors cursor-pointer" title="View Details"><Eye size={18} /></motion.button>
                      <motion.button whileHover={{ scale: 1.1 }} className="p-2 text-slate-400 hover:text-emerald-600 hover:bg-emerald-50 rounded-lg transition-colors cursor-pointer" title="Download Report"><Download size={18} /></motion.button>
                      <motion.button whileHover={{ scale: 1.1 }} className="p-2 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded-lg transition-colors cursor-pointer" title="Delete"><Trash2 size={18} /></motion.button>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </motion.tbody>
          </table>
        </div>
        
        {/* Pagination */}
        <div className="px-6 py-4 border-t border-slate-100 flex items-center justify-between bg-slate-50/50">
          <p className="text-xs font-semibold text-slate-500">Showing 1–4 of 24 videos</p>
          <div className="flex items-center gap-1.5">
            <button 
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
              className="px-3 py-1.5 text-xs font-bold bg-white border border-slate-200 text-slate-500 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              Previous
            </button>
            {[1, 2, 3, 4, 5, 6].map(page => (
              <button 
                key={page}
                onClick={() => setCurrentPage(page)}
                className={`w-8 h-8 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                  currentPage === page 
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/30' 
                    : 'bg-white border border-slate-200 text-slate-600 hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-200'
                }`}
              >
                {page}
              </button>
            ))}
            <button 
              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
              className="px-3 py-1.5 text-xs font-bold bg-white border border-slate-200 text-slate-700 rounded-lg hover:bg-slate-50 hover:text-indigo-600 transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              Next
            </button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

// Stats pill component
const StatPill = ({ label, value, icon: Icon, color, bg }) => (
  <div className={`flex items-center gap-3 px-4 py-3 rounded-2xl border ${bg} shadow-sm`}>
    <div className={`p-2 rounded-xl ${bg} ${color}`}>
      <Icon size={18} strokeWidth={2.5} />
    </div>
    <div>
      <p className={`text-xl font-black ${color}`}>{value}</p>
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{label}</p>
    </div>
  </div>
);

export default VideoLibrary;