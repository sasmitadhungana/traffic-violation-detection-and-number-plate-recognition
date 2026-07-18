import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload,
  Video,
  AlertTriangle,
  Clock,
  MoreVertical,
  PlayCircle,
  TrendingUp,
  ArrowUpRight,
  ChevronRight,
  ArrowRight,
  RefreshCw
} from "lucide-react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip as RechartsTooltip,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import '../styles/Dashboard.css';

const pieData = [
  { name: 'Speeding', value: 400 },
  { name: 'Red Light', value: 300 },
  { name: 'No Helmet', value: 300 },
  { name: 'Wrong Lane', value: 200 },
];

const COLORS = ['#6366f1', '#f43f5e', '#f59e0b', '#10b981'];

const containerVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const CustomPieTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const total = pieData.reduce((sum, d) => sum + d.value, 0);
    const pct = ((payload[0].value / total) * 100).toFixed(1);
    return (
      <div className="bg-white/95 backdrop-blur-md border border-slate-200 rounded-xl px-4 py-3 shadow-xl">
        <p className="text-sm font-bold text-slate-800">{payload[0].name}</p>
        <p className="text-xs font-semibold text-slate-500 mt-1">
          {payload[0].value} incidents · <span className="text-indigo-600 font-bold">{pct}%</span>
        </p>
      </div>
    );
  }
  return null;
};
const Dashboard = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(0);
  const [recentVideos, setRecentVideos] = useState([]);
  const [stats, setStats] = useState({ total_videos: 0, total_violations: 0, high_priority: 0, pie_data: [] });
  const [isLoading, setIsLoading] = useState(true);
  
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const fetchData = async () => {
    try {
      const [analysisRes, statsRes] = await Promise.all([
        fetch('http://localhost:8000/api/analysis?limit=5'),
        fetch('http://localhost:8000/api/stats')
      ]);
      const analysisData = await analysisRes.json();
      const statsData = await statsRes.json();
      
      setRecentVideos(analysisData);
      setStats(statsData);
      setLastUpdated(0);
    } catch (error) {
      console.error("Failed to fetch dashboard data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const refreshInterval = setInterval(() => {
      fetchData();
    }, 15000); // refresh every 15s
    return () => clearInterval(refreshInterval);
  }, []);

  // Live "last updated" counter
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdated(prev => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const handleFileUpload = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      alert(`File selected: ${e.target.files[0].name}. Redirecting to upload page...`);
      navigate('/upload-video');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      alert(`Dropped file: ${e.dataTransfer.files[0].name}. Redirecting to upload page...`);
      navigate('/upload-video');
    }
  };

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex flex-col md:flex-row md:items-end justify-between bg-white/50 p-6 rounded-3xl border border-white/60 shadow-sm backdrop-blur-md">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900 flex items-center gap-3">
            {getGreeting()}, Admin
            <div className="px-2.5 py-1 bg-indigo-50 border border-indigo-100 rounded-full text-[10px] uppercase tracking-widest text-indigo-600 font-bold flex items-center gap-1.5 shadow-sm">
              <span className="w-1.5 h-1.5 bg-indigo-500 rounded-full animate-pulse-glow"></span>
              Live Feed Active
            </div>
          </h1>
          <div className="flex items-center gap-3 mt-2">
            <p className="text-slate-500 text-sm font-medium">
              Monitor system performance, recent uploads, and detected violations in real-time.
            </p>
            <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-400">
              <RefreshCw size={12} className="animate-spin-slow" />
              Updated {lastUpdated}s ago
            </div>
          </div>
        </div>
        <div className="mt-4 md:mt-0 flex space-x-3">
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/video-library')}
            className="px-5 py-2.5 bg-white border border-slate-200 text-slate-700 text-sm font-bold rounded-xl shadow-[0_2px_10px_rgba(0,0,0,0.02)] hover:shadow-[0_4px_15px_rgba(0,0,0,0.05)] hover:border-slate-300 hover:text-slate-900 transition-all flex items-center gap-2 cursor-pointer"
          >
            <Clock size={18} className="text-slate-400" />
            History
          </motion.button>
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/upload-video')}
            className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-indigo-500 text-white text-sm font-bold rounded-xl shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/40 transition-all flex items-center gap-2 cursor-pointer relative overflow-hidden group"
          >
            <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white opacity-20 group-hover:animate-shine"></div>
            <Upload size={18} className="relative z-10" />
            <span className="relative z-10">New Upload</span>
          </motion.button>
        </div>
      </motion.div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
        <motion.div variants={itemVariants}>
          <StatCard title="Total Videos" value={isLoading ? "..." : stats.total_videos} icon={Video} color="text-indigo-600" bg="bg-indigo-50 border-indigo-100" trend="Live Updates" trendIcon={ArrowUpRight} trendColor="text-indigo-600" onClick={() => navigate('/video-library')} />
        </motion.div>
        <motion.div variants={itemVariants}>
          <StatCard title="Violations Detected" value={isLoading ? "..." : stats.total_violations} icon={AlertTriangle} color="text-rose-600" bg="bg-rose-50 border-rose-100" trend="Live Updates" trendIcon={TrendingUp} trendColor="text-rose-600" onClick={() => navigate('/reports')} />
        </motion.div>
      </div>

      {/* Main Grid */}
      <div className="space-y-6">

        {/* Quick Upload Banner */}
        <motion.div 
          variants={itemVariants}
          whileHover={{ scale: 1.005 }}
          className={`bg-white/80 backdrop-blur-md rounded-[2rem] border border-white p-2 shadow-sm transition-all duration-300 ${isDragging ? 'ring-4 ring-indigo-400/50 bg-indigo-50/50' : 'hover:shadow-md'}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <div className={`border-2 border-dashed rounded-[1.5rem] p-5 flex flex-col sm:flex-row items-center justify-between gap-6 transition-all duration-300 cursor-pointer ${isDragging ? 'border-indigo-400 bg-indigo-50/30 scale-[0.99]' : 'border-slate-200 bg-slate-50/30 hover:bg-slate-50'}`} onClick={() => fileInputRef.current?.click()}>
            <div className="flex items-center gap-5">
              <motion.div 
                animate={isDragging ? { y: [0, -10, 0], scale: 1.1 } : {}}
                transition={{ repeat: isDragging ? Infinity : 0, duration: 1 }}
                className="h-14 w-14 bg-white rounded-2xl shadow-sm flex items-center justify-center text-indigo-500 border border-slate-100"
              >
                <Upload size={24} />
              </motion.div>
              <div>
                <h3 className="text-base font-extrabold text-slate-800">Quick Upload</h3>
                <p className="text-sm text-slate-500 font-medium mt-0.5">{isDragging ? 'Drop it right here!' : 'Drag & drop a video file here, or click to browse (Max 500MB)'}</p>
              </div>
            </div>
            <motion.button 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
              className="bg-white border border-slate-200 px-6 py-2.5 rounded-xl text-sm font-bold text-slate-700 shadow-sm hover:border-indigo-300 hover:text-indigo-700 transition-colors w-full sm:w-auto cursor-pointer flex items-center gap-2"
            >
              Browse Files
            </motion.button>
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              accept="video/*" 
              onChange={handleFileUpload}
            />
          </div>
        </motion.div>

        <div className="space-y-6">

          {/* Recent Videos Table */}
          <motion.div variants={itemVariants} className="bg-white/90 backdrop-blur-xl rounded-[2rem] border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden">
            <div className="p-6 lg:p-8 border-b border-slate-100 flex justify-between items-center bg-white/50">
              <div>
                <h2 className="text-xl font-extrabold text-slate-800">Recent Analysis</h2>
                <p className="text-sm text-slate-500 mt-1 font-medium">Latest processed footage</p>
              </div>
              <button 
                onClick={() => navigate('/video-library')}
                className="text-sm font-bold text-indigo-600 hover:text-indigo-700 bg-indigo-50 hover:bg-indigo-100 px-4 py-2 rounded-xl transition-colors cursor-pointer flex items-center gap-1.5"
              >
                View All <ChevronRight size={16} />
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50/50 border-b border-slate-100 text-xs uppercase tracking-widest text-slate-500 font-bold">
                    <th className="px-6 py-4">Video Name</th>
                    <th className="px-6 py-4">Date</th>
                    <th className="px-6 py-4">Status</th>
                    <th className="px-6 py-4 text-center">Violations</th>
                    <th className="px-6 py-4"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100/50">
                  {recentVideos.length > 0 ? (
                    recentVideos.map((video) => (
                      <TableRow 
                        key={video.id} 
                        name={video.filename} 
                        date={new Date(video.upload_time).toLocaleString()} 
                        status={video.status} 
                        count={video.violation_count || "-"} 
                        navigate={navigate} 
                      />
                    ))
                  ) : (
                    <tr>
                      <td colSpan="5" className="px-6 py-8 text-center text-slate-400 font-medium">
                        No recent videos found. Upload a video to get started!
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>

      </div>
    </motion.div>
  );
};;

// Helper Components
const StatCard = ({ title, value, icon: Icon, color, bg, trend, trendIcon: TrendIcon, trendColor, onClick }) => (
  <motion.div 
    whileHover={{ y: -5, scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    transition={{ type: "spring", stiffness: 400, damping: 25 }}
    onClick={onClick} 
    className={`bg-white/90 backdrop-blur-xl rounded-[2rem] p-6 lg:p-8 border border-white shadow-[0_8px_30px_rgb(0,0,0,0.04)] relative overflow-hidden group hover:shadow-[0_12px_40px_rgb(0,0,0,0.08)] cursor-pointer`}
  >
    <div className="flex justify-between items-start mb-6 relative z-10">
      <div className={`p-3.5 rounded-2xl ${bg} ${color} border shadow-inner`}>
        <Icon size={24} strokeWidth={2.5} />
      </div>
      <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 bg-slate-100/80 px-3 py-1.5 rounded-lg">Today</span>
    </div>
    <div className="relative z-10">
      <h3 className="text-4xl font-black text-slate-900 tracking-tight mb-2 group-hover:scale-[1.02] transform origin-left transition-transform flex items-center gap-2">
        {value}
      </h3>
      <p className="text-slate-500 font-bold text-sm">{title}</p>
    </div>
    <div className="mt-6 pt-5 border-t border-slate-100 relative z-10 flex items-center gap-2.5">
      <div className={`p-1 rounded-md ${bg}`}>
        {TrendIcon && <TrendIcon size={14} className={trendColor} strokeWidth={3} />}
      </div>
      <p className={`text-xs font-bold ${trendColor || 'text-slate-500'}`}>{trend}</p>
    </div>
    <div className={`absolute -right-6 -bottom-6 opacity-[0.03] transition-transform duration-500 group-hover:scale-110 group-hover:rotate-12 ${color} z-0`}>
      <Icon size={140} />
    </div>
  </motion.div>
);

const TableRow = ({ name, date, status, count, navigate }) => (
  <tr className="group text-sm cursor-pointer hover:bg-slate-50/80 transition-colors" onClick={() => navigate('/video-library')}>
    <td className="px-6 py-4">
      <div className="flex items-center gap-4">
        <motion.div whileHover={{ scale: 1.1 }} className="p-2 bg-indigo-50 text-indigo-600 rounded-xl group-hover:bg-indigo-100 shadow-sm">
          <PlayCircle size={20} />
        </motion.div>
        <span className="font-bold text-slate-800">{name}</span>
      </div>
    </td>
    <td className="px-6 py-4 text-slate-500 font-semibold">{date}</td>
    <td className="px-6 py-4">
      <span className={`px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider flex w-fit items-center gap-2 ${
        status === 'Completed' ? 'bg-emerald-50 text-emerald-700 border border-emerald-100 shadow-sm' : 'bg-blue-50 text-blue-700 border border-blue-100 shadow-sm'
      }`}>
        <span className={`w-1.5 h-1.5 rounded-full ${status === 'Completed' ? 'bg-emerald-500' : 'bg-blue-500 animate-pulse'}`}></span>
        {status}
      </span>
    </td>
    <td className="px-6 py-4 text-center">
      {status === 'Completed' ? (
        <span className="font-extrabold text-rose-600 bg-rose-50 border border-rose-100 px-3 py-1 rounded-lg">{count}</span>
      ) : (
        <span className="text-slate-300 font-bold">-</span>
      )}
    </td>
    <td className="px-6 py-4 text-right">
      <button 
        onClick={(e) => { e.stopPropagation(); navigate('/video-library'); }}
        className="p-2 text-slate-400 hover:text-indigo-600 rounded-xl hover:bg-indigo-50 transition-colors cursor-pointer"
      >
        <MoreVertical size={18} />
      </button>
    </td>
  </tr>
);

const FeedItem = ({ type, time, cam, dotColor }) => (
  <motion.div 
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    exit={{ opacity: 0, scale: 0.9 }}
    className="flex gap-4 items-start group cursor-pointer hover:bg-slate-50/80 p-3 rounded-2xl transition-colors relative z-10"
  >
    <div className="relative flex flex-col items-center mt-1">
      <div className={`w-3.5 h-3.5 rounded-full ${dotColor} ring-4 ring-white shadow-sm z-10`}></div>
    </div>
    <div className="pb-1 w-full">
      <div className="flex justify-between items-start">
        <p className="font-bold text-slate-800 text-sm flex items-center gap-2">
          {type}
        </p>
        <span className="text-[10px] font-bold text-slate-400 bg-white border border-slate-100 shadow-sm px-2 py-1 rounded-lg">{time}</span>
      </div>
      <div className="flex items-center gap-1.5 mt-1.5 text-xs font-semibold text-slate-500">
        <Video size={12} className="text-slate-400" />
        <span>{cam}</span>
      </div>
    </div>
  </motion.div>
);

export default Dashboard;