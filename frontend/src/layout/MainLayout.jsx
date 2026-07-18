import Sidebar from "./Sidebar";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { Bell, Search, Calendar, ChevronDown, ChevronRight, Home } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useMemo } from "react";

const pageTitles = {
  "/": "Dashboard",
  "/upload-video": "Upload Video",
  "/video-library": "Video Library",
  "/reports": "Reports",
  "/settings": "Settings",
};

const MainLayout = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Dynamic date
  const formattedDate = useMemo(() => {
    const now = new Date();
    return now.toLocaleDateString('en-US', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
    });
  }, []);

  const dayOfWeek = useMemo(() => {
    return new Date().toLocaleDateString('en-US', { weekday: 'long' });
  }, []);

  const currentPage = pageTitles[location.pathname] || "Page";

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden font-sans relative">
      
      {/* Premium Animated Background Elements */}
      <div className="absolute top-[-10%] left-[-5%] w-[40%] h-[40%] bg-indigo-400/10 rounded-full blur-[120px] pointer-events-none animate-float"></div>
      <div className="absolute bottom-[-10%] right-[-5%] w-[30%] h-[40%] bg-purple-400/10 rounded-full blur-[100px] pointer-events-none animate-float delay-500"></div>

      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden relative z-10">

        {/* Top Header */}
        <header className="h-[72px] bg-white/70 backdrop-blur-xl border-b border-white/50 flex items-center justify-between px-8 sticky top-0 z-20 shadow-[0_4px_30px_rgba(0,0,0,0.02)]">

          <div className="flex items-center gap-6">
            {/* Breadcrumb */}
            <nav className="hidden md:flex items-center gap-2 text-sm">
              <button 
                onClick={() => navigate('/')} 
                className="text-slate-400 hover:text-indigo-600 transition-colors cursor-pointer flex items-center gap-1"
              >
                <Home size={14} />
                <span className="font-semibold">Home</span>
              </button>
              <ChevronRight size={14} className="text-slate-300" />
              <span className="font-bold text-slate-700">{currentPage}</span>
            </nav>

            {/* Global Search */}
            <div className="relative group">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors" size={18} />
              <input 
                type="text" 
                placeholder="Search vehicles, cameras, or violations..." 
                className="w-full min-w-[280px] lg:min-w-[360px] bg-slate-100/50 hover:bg-slate-100 focus:bg-white border border-transparent focus:border-indigo-200 pl-10 pr-4 py-2 rounded-xl text-sm transition-all focus:outline-none focus:ring-4 focus:ring-indigo-500/10 font-medium"
              />
            </div>
          </div>

          <div className="flex items-center gap-5 ml-4">
            
            {/* Notification Bell */}
            <motion.button 
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="relative p-2 text-slate-400 hover:text-slate-700 transition-colors rounded-xl hover:bg-slate-100 cursor-pointer"
            >
              <Bell size={20} />
              <span className="absolute top-1 right-1.5 w-2 h-2 bg-rose-500 rounded-full animate-pulse-glow"></span>
            </motion.button>

            <div className="h-6 w-[1px] bg-slate-200"></div>

            {/* Date Card */}
            <div className="flex items-center gap-2 text-slate-600 bg-slate-50 px-3 py-1.5 rounded-xl border border-slate-100">
              <Calendar size={16} className="text-slate-400" />
              <div>
                <p className="text-xs font-bold leading-none text-slate-700">
                  {formattedDate}
                </p>
                <p className="text-[10px] font-semibold text-slate-400 mt-0.5">
                  {dayOfWeek}
                </p>
              </div>
            </div>

            {/* User Dropdown */}
            <motion.div 
              whileHover={{ y: -2 }}
              onClick={() => navigate('/settings')}
              className="flex items-center gap-3 pl-2 cursor-pointer group"
            >
              <motion.div 
                whileHover={{ rotate: 10, scale: 1.1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
                className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold text-sm shadow-md group-hover:shadow-lg group-hover:shadow-indigo-500/30 transition-all"
              >
                A
              </motion.div>
              <div className="hidden md:block">
                <p className="text-sm font-bold text-slate-700 leading-none">Admin User</p>
                <p className="text-[11px] font-semibold text-slate-400 mt-0.5">System Administrator</p>
              </div>
              <ChevronDown size={16} className="text-slate-400 group-hover:text-slate-600 transition-colors" />
            </motion.div>

          </div>

        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-6 lg:p-8 relative">
          <div className="max-w-7xl mx-auto w-full relative z-10">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 20, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.98 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </div>
        </main>

      </div>

    </div>
  );
};

export default MainLayout;