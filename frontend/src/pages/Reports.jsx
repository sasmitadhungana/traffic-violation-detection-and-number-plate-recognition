import React, { useState } from "react";
import { FileText, Download, Calendar, FileSpreadsheet, Search, CheckCircle2, Loader2, FileBarChart } from "lucide-react";
import { motion } from "framer-motion";

const Reports = () => {
  const [downloadingId, setDownloadingId] = useState(null);

  const [reports, setReports] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  React.useEffect(() => {
    fetch('http://localhost:8000/api/reports')
      .then(res => res.json())
      .then(data => {
        const mappedReports = data.map((v, i) => ({
          id: `VIOL-${v.video_id.substring(0,6).toUpperCase()}-${i}`,
          name: v.type,
          date: new Date(v.timestamp).toLocaleString(),
          type: "LOG",
          size: `${(v.confidence * 100).toFixed(1)}% confidence`,
          original: v
        }));
        setReports(mappedReports);
        setIsLoading(false);
      })
      .catch(err => {
        console.error("Error fetching reports:", err);
        setIsLoading(false);
      });
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.08 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  const handleDownload = (id) => {
    setDownloadingId(id);
    setTimeout(() => {
      setDownloadingId(null);
    }, 1500);
  };

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex flex-col md:flex-row md:items-end justify-between bg-white/50 p-6 rounded-[2rem] border border-white/60 shadow-sm backdrop-blur-md">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Reports</h1>
          <p className="text-slate-500 font-medium text-sm mt-1.5">Generate and download compliance reports</p>
        </div>
        <div className="flex gap-3 mt-4 md:mt-0">
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-5 py-2.5 bg-white border border-slate-200 text-slate-700 rounded-xl text-sm font-bold shadow-sm hover:border-indigo-300 hover:text-indigo-600 transition-all flex items-center gap-2 cursor-pointer"
          >
            <Calendar size={18} /> Select Date Range
          </motion.button>
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-indigo-500 text-white rounded-xl text-sm font-bold shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/40 transition-all flex items-center gap-2 relative overflow-hidden group cursor-pointer"
          >
            <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white opacity-20 group-hover:animate-shine"></div>
            <FileBarChart size={18} className="relative z-10" /> <span className="relative z-10">Generate New Report</span>
          </motion.button>
        </div>
      </motion.div>

      {/* Stats Bar */}
      <motion.div variants={itemVariants} className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatPill label="Total Reports" value={reports.length} icon={FileBarChart} color="text-indigo-600" bg="bg-indigo-50 border-indigo-100" />
        <StatPill label="Generated This Week" value="12" icon={Calendar} color="text-emerald-600" bg="bg-emerald-50 border-emerald-100" />
        <StatPill label="Scheduled" value="3" icon={Loader2} color="text-amber-600" bg="bg-amber-50 border-amber-100" />
        <StatPill label="Storage Used" value="45 MB" icon={FileText} color="text-slate-600" bg="bg-slate-50 border-slate-100" />
      </motion.div>

      {/* Reports Table */}
      <motion.div variants={itemVariants} className="bg-white/80 backdrop-blur-xl rounded-[2rem] border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden">
        <div className="p-6 lg:p-8 border-b border-slate-100 flex justify-between items-center bg-white/50">
          <div>
            <h2 className="text-xl font-extrabold text-slate-800">Available Downloads</h2>
            <p className="text-sm text-slate-400 mt-1 font-medium">{reports.length} reports available</p>
          </div>
          <div className="relative group">
            <Search size={18} className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
            <input type="text" placeholder="Search reports..." className="pl-12 pr-4 py-2.5 border border-slate-200 rounded-xl text-sm font-semibold text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 bg-white transition-all w-64 shadow-sm" />
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left whitespace-nowrap">
            <thead>
              <tr className="bg-slate-50/50 border-b border-slate-100 text-xs uppercase tracking-widest text-slate-500 font-bold">
                <th className="px-6 py-5">Report Name</th>
                <th className="px-6 py-5">Date Generated</th>
                <th className="px-6 py-5">Format</th>
                <th className="px-6 py-5 text-right">Action</th>
              </tr>
            </thead>
            <motion.tbody 
              variants={containerVariants}
              initial="hidden"
              animate="show"
              className="divide-y divide-slate-100/50"
            >
              {reports.map((report) => (
                <motion.tr 
                  variants={itemVariants}
                  key={report.id} 
                  className="hover:bg-indigo-50/30 transition-all duration-200 group text-sm relative"
                >
                  <td className="px-6 py-5 font-bold text-slate-800 flex items-center gap-4 group-hover:text-indigo-600 transition-colors cursor-pointer relative">
                    <div className="absolute left-0 top-2 bottom-2 w-1 bg-indigo-500 rounded-r-full opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
                    <div className={`p-3 rounded-xl transition-all duration-300 ${report.type === 'PDF' ? 'bg-rose-50 text-rose-600 group-hover:bg-rose-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-rose-500/20' : 'bg-emerald-50 text-emerald-600 group-hover:bg-emerald-600 group-hover:text-white group-hover:shadow-lg group-hover:shadow-emerald-500/20'}`}>
                      {report.type === 'PDF' ? <FileText size={20} /> : <FileSpreadsheet size={20} />}
                    </div>
                    <div>
                      <span>{report.name}</span>
                      <p className="text-[10px] font-semibold text-slate-400 mt-0.5 uppercase tracking-wider">{report.id}</p>
                    </div>
                  </td>
                  <td className="px-6 py-5 text-slate-500 font-semibold">{report.date}</td>
                  <td className="px-6 py-5">
                    <span className={`px-3 py-1.5 rounded-lg text-xs font-bold shadow-sm border ${
                      report.type === 'PDF' 
                        ? 'bg-rose-50 text-rose-600 border-rose-100' 
                        : 'bg-emerald-50 text-emerald-600 border-emerald-100'
                    }`}>
                      {report.type} &bull; {report.size}
                    </span>
                  </td>
                  <td className="px-6 py-5 text-right">
                    <motion.button 
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleDownload(report.id)}
                      disabled={downloadingId === report.id}
                      className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-xl transition-all shadow-sm cursor-pointer ${
                        downloadingId === report.id
                          ? 'bg-emerald-600 text-white shadow-emerald-500/30'
                          : 'text-indigo-600 hover:text-white hover:bg-indigo-600 bg-indigo-50 hover:shadow-indigo-500/30'
                      }`}
                    >
                      {downloadingId === report.id ? (
                        <>
                          <CheckCircle2 size={16} className="animate-scale-in" /> Downloaded
                        </>
                      ) : (
                        <>
                          <Download size={16} /> Download
                        </>
                      )}
                    </motion.button>
                  </td>
                </motion.tr>
              ))}
            </motion.tbody>
          </table>
        </div>

        {/* Empty state (hidden when reports exist, shown as fallback) */}
        {reports.length === 0 && (
          <div className="p-16 text-center flex flex-col items-center justify-center">
            <div className="w-20 h-20 bg-slate-100 rounded-3xl flex items-center justify-center mb-5">
              <FileBarChart size={36} className="text-slate-400" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2">No Reports Generated</h3>
            <p className="text-slate-500 text-sm max-w-sm">Generate your first compliance report to see it listed here.</p>
          </div>
        )}
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

export default Reports;