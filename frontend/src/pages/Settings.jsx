import React, { useState } from "react";
import { 
  Save, 
  Bell, 
  Shield, 
  Sliders, 
  User, 
  Key, 
  Database,
  CheckCircle2,
  Mail,
  Smartphone,
  Server,
  Upload as UploadIcon
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const Settings = () => {
  const [activeTab, setActiveTab] = useState("analysis");
  const [isSaving, setIsSaving] = useState(false);
  const [confidence, setConfidence] = useState(85);
  const [toastMessage, setToastMessage] = useState("");

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      setIsSaving(false);
      setToastMessage("Settings saved successfully!");
      setTimeout(() => setToastMessage(""), 3000);
    }, 1000);
  };

  const tabs = [
    { id: "profile", name: "My Profile", icon: User },
    { id: "analysis", name: "Analysis Engine", icon: Sliders },
    { id: "notifications", name: "Notifications", icon: Bell },
    { id: "security", name: "Security & Access", icon: Shield },
    { id: "api", name: "API & Integrations", icon: Key },
    { id: "system", name: "System Logs", icon: Database },
  ];

  const tabContentVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.3, ease: "easeOut" } },
    exit: { opacity: 0, y: -10, transition: { duration: 0.15 } }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6 animate-fade-in-up pb-10"
    >
      
      {/* Floating Toast */}
      <AnimatePresence>
        {toastMessage && (
          <motion.div
            initial={{ opacity: 0, y: -20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.95 }}
            className="fixed top-6 right-6 z-50 bg-emerald-600 text-white px-5 py-3 rounded-2xl shadow-xl shadow-emerald-500/30 flex items-center gap-2.5 font-bold text-sm"
          >
            <CheckCircle2 size={18} />
            {toastMessage}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between bg-white/50 p-6 rounded-[2rem] border border-white/60 shadow-sm backdrop-blur-md">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
            System Settings
          </h1>
          <p className="text-slate-500 mt-1.5 text-sm font-medium">
            Manage your account, detection thresholds, and platform integrations.
          </p>
        </div>
        <div className="flex items-center gap-3 mt-4 md:mt-0">
          <motion.button 
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleSave}
            disabled={isSaving}
            className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-indigo-500 text-white rounded-xl text-sm font-bold shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/40 transition-all flex items-center gap-2 relative overflow-hidden group disabled:opacity-70 disabled:cursor-not-allowed cursor-pointer"
          >
            <div className="absolute top-0 -inset-full h-full w-1/2 z-5 block transform -skew-x-12 bg-gradient-to-r from-transparent to-white opacity-20 group-hover:animate-shine"></div>
            {isSaving ? (
              <svg className="animate-spin h-4 w-4 text-white relative z-10" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <Save size={18} className="relative z-10" />
            )}
            <span className="relative z-10">{isSaving ? "Saving..." : "Save Changes"}</span>
          </motion.button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 lg:gap-8 mt-8">
        
        {/* Sidebar Navigation */}
        <div className="lg:col-span-1 space-y-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <motion.button
                key={tab.id}
                whileHover={{ x: isActive ? 0 : 4 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full text-left p-4 rounded-xl transition-all flex items-center gap-3 font-semibold text-[15px] cursor-pointer ${
                  isActive 
                    ? "bg-white border border-indigo-100 shadow-sm text-indigo-700 relative overflow-hidden" 
                    : "border border-transparent text-slate-500 hover:bg-white hover:text-slate-800 hover:shadow-sm hover:border-slate-200"
                }`}
              >
                {isActive && (
                  <motion.div 
                    layoutId="settings-active-tab"
                    className="absolute left-0 top-0 bottom-0 w-1 bg-indigo-600 rounded-r-full"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                  />
                )}
                <Icon size={20} className={isActive ? "text-indigo-600" : "text-slate-400"} /> 
                {tab.name}
              </motion.button>
            );
          })}
        </div>

        {/* Main Settings Area */}
        <div className="lg:col-span-3">
          <div className="bg-white rounded-[2rem] border border-slate-200 shadow-sm overflow-hidden transition-all duration-300">
            
            <AnimatePresence mode="wait">
              {/* Analysis Engine Tab */}
              {activeTab === "analysis" && (
                <motion.div key="analysis" variants={tabContentVariants} initial="hidden" animate="visible" exit="exit">
                  <div className="p-6 md:p-8 border-b border-slate-100">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
                        <Sliders size={20} />
                      </div>
                      <h2 className="text-xl font-bold text-slate-800">Analysis Engine Configuration</h2>
                    </div>
                    <p className="text-sm text-slate-500 font-medium">Fine-tune the AI detection parameters and active models.</p>
                  </div>
                  
                  <div className="p-6 md:p-8 space-y-10">
                    
                    {/* Threshold Slider */}
                    <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100">
                      <div className="flex items-center justify-between mb-2">
                        <div>
                          <span className="font-bold text-slate-800">Global Confidence Threshold</span>
                          <p className="text-xs text-slate-500 font-medium mt-1">Minimum AI confidence required to flag a violation automatically.</p>
                        </div>
                        <div className="bg-white px-4 py-2 rounded-xl border border-indigo-100 shadow-sm">
                          <span className="text-indigo-600 font-extrabold text-xl">{confidence}%</span>
                        </div>
                      </div>
                      
                      <div className="mt-6 relative">
                        <input 
                          type="range" 
                          min="50" max="99" 
                          value={confidence}
                          onChange={(e) => setConfidence(e.target.value)}
                          className="w-full cursor-pointer" 
                        />
                        <div className="flex justify-between text-[10px] font-bold text-slate-400 mt-2 uppercase tracking-wider">
                          <span>Lenient (50%)</span>
                          <span>Strict (99%)</span>
                        </div>
                      </div>
                    </div>

                    {/* Modules */}
                    <div>
                      <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <Server size={18} className="text-slate-400" />
                        Active Violation Models
                      </h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <ToggleCard 
                          title="Speeding Detection" 
                          desc="Monitors vehicle speed against posted limits via camera calibration."
                          defaultOn={false}
                          badge="FUTURE"
                        />
                        <ToggleCard 
                          title="Red Light Running" 
                          desc="Detects vehicles crossing the intersection line during red phases."
                          defaultOn={true}
                        />
                        <ToggleCard 
                          title="Helmet Detection" 
                          desc="Identifies two-wheeler riders without safety helmets."
                          defaultOn={false}
                          badge="FUTURE"
                        />
                        <ToggleCard 
                          title="Wrong-Way Driving" 
                          desc="Alerts on vehicles traveling against the designated flow of traffic."
                          defaultOn={false}
                          badge="FUTURE"
                        />
                        <ToggleCard 
                          title="Illegal Parking" 
                          desc="Detects vehicles stationary in no-parking zones for extended periods."
                          defaultOn={false}
                          badge="FUTURE"
                        />
                        <ToggleCard 
                          title="Mobile Phone Usage" 
                          desc="Beta: Detects drivers using handheld devices."
                          defaultOn={false}
                          badge="FUTURE"
                        />
                      </div>

                    </div>

                  </div>
                </motion.div>
              )}

              {/* Profile Tab */}
              {activeTab === "profile" && (
                <motion.div key="profile" variants={tabContentVariants} initial="hidden" animate="visible" exit="exit">
                  <div className="p-6 md:p-8 border-b border-slate-100">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="p-2 bg-blue-50 text-blue-600 rounded-lg">
                        <User size={20} />
                      </div>
                      <h2 className="text-xl font-bold text-slate-800">My Profile</h2>
                    </div>
                    <p className="text-sm text-slate-500 font-medium">Update your personal information and preferences.</p>
                  </div>
                  
                  <div className="p-6 md:p-8 space-y-6">
                    <div className="flex items-center gap-6">
                      <div className="relative group cursor-pointer">
                        <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-indigo-100 to-purple-100 border-2 border-white shadow-md flex items-center justify-center text-indigo-600 font-bold text-3xl">
                          A
                        </div>
                        <div className="absolute inset-0 bg-slate-900/40 rounded-2xl opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                          <UploadIcon size={24} className="text-white" />
                        </div>
                      </div>
                      <div>
                        <button className="px-4 py-2 bg-white border border-slate-200 text-slate-700 text-sm font-semibold rounded-xl shadow-sm hover:bg-slate-50 hover:border-indigo-200 hover:text-indigo-600 transition-all cursor-pointer">
                          Change Avatar
                        </button>
                        <p className="text-xs text-slate-500 mt-2 font-medium">JPG, GIF or PNG. Max size of 800K</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
                      <InputField label="First Name" defaultValue="System" />
                      <InputField label="Last Name" defaultValue="Admin" />
                      <InputField label="Email Address" defaultValue="admin@trafficguard.com" type="email" />
                      <InputField label="Phone Number" defaultValue="+1 (555) 000-0000" type="tel" />
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Notifications Tab */}
              {activeTab === "notifications" && (
                <motion.div key="notifications" variants={tabContentVariants} initial="hidden" animate="visible" exit="exit">
                  <div className="p-6 md:p-8 border-b border-slate-100">
                    <div className="flex items-center gap-3 mb-2">
                      <div className="p-2 bg-amber-50 text-amber-600 rounded-lg">
                        <Bell size={20} />
                      </div>
                      <h2 className="text-xl font-bold text-slate-800">Alerts & Notifications</h2>
                    </div>
                    <p className="text-sm text-slate-500 font-medium">Choose how and when you want to be notified.</p>
                  </div>
                  
                  <div className="p-6 md:p-8 space-y-8">
                    <div>
                      <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <Mail size={18} className="text-slate-400" />
                        Email Notifications
                      </h3>
                      <div className="space-y-4">
                        <NotificationRow title="Daily Summary" desc="Receive a daily digest of all violations processed." defaultOn={true} />
                        <NotificationRow title="System Errors" desc="Get alerted when a camera feed drops or analysis fails." defaultOn={true} />
                        <NotificationRow title="High-Priority Alerts" desc="Immediate email for severe violations (e.g., extreme speeding)." defaultOn={false} />
                      </div>
                    </div>

                    <div className="pt-4 border-t border-slate-100">
                      <h3 className="font-bold text-slate-800 mb-4 flex items-center gap-2">
                        <Smartphone size={18} className="text-slate-400" />
                        Push & SMS Alerts
                      </h3>
                      <div className="space-y-4">
                        <NotificationRow title="System Offline" desc="SMS alert if the entire backend cluster goes down." defaultOn={true} />
                        <NotificationRow title="Weekly Reports" desc="Push notification when your weekly analytics are ready." defaultOn={false} />
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Placeholder for other tabs */}
              {["security", "api", "system"].includes(activeTab) && (
                <motion.div key="placeholder" variants={tabContentVariants} initial="hidden" animate="visible" exit="exit" className="p-16 text-center flex flex-col items-center justify-center min-h-[400px]">
                  <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mb-4">
                    <Shield size={28} className="text-slate-400" />
                  </div>
                  <h3 className="text-xl font-bold text-slate-800 mb-2">Advanced Configuration</h3>
                  <p className="text-slate-500 max-w-md mx-auto text-sm">
                    This section requires super-admin privileges. Please contact your infrastructure provider to unlock these endpoints.
                  </p>
                  <button className="mt-6 px-5 py-2.5 bg-white border border-slate-200 text-slate-700 text-sm font-bold rounded-xl shadow-sm hover:bg-slate-50 hover:border-indigo-200 hover:text-indigo-600 transition-all cursor-pointer">
                    Request Access
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

          </div>
        </div>
      </div>
    </motion.div>
  );
};

// --- Helper Components ---

const ToggleCard = ({ title, desc, defaultOn, badge }) => {
  const [isOn, setIsOn] = useState(defaultOn);
  return (
    <motion.div 
      whileHover={{ y: -2 }}
      onClick={() => setIsOn(!isOn)}
      className={`p-5 border rounded-2xl transition-all cursor-pointer flex flex-col justify-between ${
        isOn ? 'border-indigo-200 bg-indigo-50/30 shadow-sm' : 'border-slate-200 bg-white hover:border-slate-300'
      }`}
    >
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <h4 className={`font-bold ${isOn ? 'text-indigo-900' : 'text-slate-700'}`}>{title}</h4>
            {badge && (
              <span className="px-2 py-0.5 rounded-md text-[9px] font-extrabold uppercase tracking-wider bg-amber-100 text-amber-700 border border-amber-200">
                {badge}
              </span>
            )}
          </div>
          {/* iOS Style Toggle */}
          <div className={`w-11 h-6 flex items-center rounded-full p-1 transition-colors duration-300 ${isOn ? 'bg-indigo-600' : 'bg-slate-300'}`}>
            <motion.div 
              animate={{ x: isOn ? 20 : 0 }}
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
              className="bg-white w-4 h-4 rounded-full shadow-md"
            />
          </div>
        </div>
        <p className={`text-xs font-medium leading-relaxed ${isOn ? 'text-indigo-700/70' : 'text-slate-500'}`}>
          {desc}
        </p>
      </div>
    </motion.div>
  );
};

const NotificationRow = ({ title, desc, defaultOn }) => {
  const [isOn, setIsOn] = useState(defaultOn);
  return (
    <div className="flex items-start justify-between p-4 rounded-2xl border border-slate-100 bg-slate-50/50 hover:bg-slate-50 transition-colors">
      <div className="pr-4">
        <h4 className="font-semibold text-slate-800 text-sm">{title}</h4>
        <p className="text-xs text-slate-500 font-medium mt-0.5">{desc}</p>
      </div>
      <div 
        onClick={() => setIsOn(!isOn)}
        className={`w-11 h-6 flex items-center rounded-full p-1 transition-colors duration-300 cursor-pointer shrink-0 mt-1 ${isOn ? 'bg-emerald-500' : 'bg-slate-300'}`}
      >
        <motion.div 
          animate={{ x: isOn ? 20 : 0 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
          className="bg-white w-4 h-4 rounded-full shadow-md"
        />
      </div>
    </div>
  );
};

const InputField = ({ label, defaultValue, type = "text" }) => (
  <div>
    <label className="block text-sm font-semibold text-slate-700 mb-1.5">
      {label}
    </label>
    <input
      type={type}
      defaultValue={defaultValue}
      className="block w-full px-4 py-2.5 border border-slate-200 rounded-xl text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all sm:text-sm font-medium bg-slate-50 focus:bg-white shadow-sm hover:border-slate-300"
    />
  </div>
);

export default Settings;