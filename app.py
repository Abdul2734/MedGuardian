import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import { Heart, Activity, AlertTriangle, TrendingUp, User, Calendar, Bell, Settings } from 'lucide-react';

const MedGuardianDashboard = () => {
  const [user, setUser] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  // Mock data for demonstration
  const mockUser = {
    id: 1,
    name: "John Doe",
    email: "john.doe@email.com",
    age: 45,
    lastAssessment: "2024-01-15"
  };

  const mockDashboardData = {
    summary: {
      totalAssessments: 12,
      highRiskConditions: 2,
      criticalRiskConditions: 0,
      lastAssessment: "2024-01-15T10:30:00Z"
    },
    riskAssessments: [
      {
        id: 1,
        diseaseType: "Cardiovascular",
        riskScore: 68,
        riskLevel: "medium",
        factors: ["Age", "High Blood Pressure", "Family History"],
        recommendations: ["Regular exercise", "Monitor blood pressure", "Dietary changes"]
      },
      {
        id: 2,
        diseaseType: "Diabetes",
        riskScore: 45,
        riskLevel: "low",
        factors: ["BMI", "Sedentary lifestyle"],
        recommendations: ["Increase physical activity", "Maintain healthy weight"]
      },
      {
        id: 3,
        diseaseType: "Cancer",
        riskScore: 25,
        riskLevel: "low",
        factors: ["Age"],
        recommendations: ["Regular screenings", "Healthy lifestyle"]
      }
    ],
    vitalsHistory: [
      { date: '2024-01-01', heartRate: 72, bloodPressure: 125, temperature: 98.6 },
      { date: '2024-01-02', heartRate: 75, bloodPressure: 128, temperature: 98.4 },
      { date: '2024-01-03', heartRate: 70, bloodPressure: 122, temperature: 98.7 },
      { date: '2024-01-04', heartRate: 73, bloodPressure: 126, temperature: 98.5 },
      { date: '2024-01-05', heartRate: 71, bloodPressure: 124, temperature: 98.6 },
      { date: '2024-01-06', heartRate: 74, bloodPressure: 129, temperature: 98.3 },
      { date: '2024-01-07', heartRate: 69, bloodPressure: 121, temperature: 98.8 }
    ]
  };

  useEffect(() => {
    // Simulate API call
    setTimeout(() => {
      setUser(mockUser);
      setDashboardData(mockDashboardData);
      setLoading(false);
    }, 1000);
  }, []);

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'low': return '#10B981';
      case 'medium': return '#F59E0B';
      case 'high': return '#EF4444';
      case 'critical': return '#DC2626';
      default: return '#6B7280';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'low': return <Heart className="w-5 h-5 text-green-500" />;
      case 'medium': return <Activity className="w-5 h-5 text-yellow-500" />;
      case 'high': return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'critical': return <AlertTriangle className="w-5 h-5 text-red-600" />;
      default: return <Heart className="w-5 h-5 text-gray-500" />;
    }
  };

  const TabButton = ({ id, label, icon, active, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
        active
          ? 'bg-blue-600 text-white'
          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  );

  const RiskCard = ({ assessment }) => (
    <div className="bg-white rounded-lg shadow-md p-6 border-l-4" style={{ borderLeftColor: getRiskColor(assessment.riskLevel) }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          {getRiskIcon(assessment.riskLevel)}
          <h3 className="text-lg font-semibold text-gray-900">{assessment.diseaseType}</h3>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold" style={{ color: getRiskColor(assessment.riskLevel) }}>
            {assessment.riskScore}%
          </div>
          <div className="text-sm text-gray-500 capitalize">{assessment.riskLevel} Risk</div>
        </div>
      </div>
      
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Risk Factors:</h4>
        <div className="flex flex-wrap gap-2">
          {assessment.factors.map((factor, index) => (
            <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full">
              {factor}
            </span>
          ))}
        </div>
      </div>
      
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">Recommendations:</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          {assessment.recommendations.slice(0, 2).map((rec, index) => (
            <li key={index} className="flex items-start space-x-2">
              <span className="text-blue-500 mt-1">•</span>
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );

  const StatCard = ({ title, value, icon, trend, color = "blue" }) => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {trend && (
            <div className="flex items-center mt-2">
              <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
              <span className="text-sm text-green-600">{trend}</span>
            </div>
          )}
        </div>
        <div className={`p-3 rounded-full bg-${color}-100`}>
          {React.cloneElement(icon, { className: `w-8 h-8 text-${color}-600` })}
        </div>
      </div>
    </div>
  );

  const VitalsChart = () => (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Vital Signs Trend</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={dashboardData?.vitalsHistory}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="heartRate" stroke="#8884d8" strokeWidth={2} name="Heart Rate" />
          <Line type="monotone" dataKey="bloodPressure" stroke="#82ca9d" strokeWidth={2} name="Blood Pressure" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );

  const RiskDistributionChart = () => {
    const riskData = dashboardData?.riskAssessments.reduce((acc, assessment) => {
      acc[assessment.riskLevel] = (acc[assessment.riskLevel] || 0) + 1;
      return acc;
    }, {});

    const chartData = Object.entries(riskData || {}).map(([level, count]) => ({
      name: level.charAt(0).toUpperCase() + level.slice(1),
      value: count,
      color: getRiskColor(level)
    }));

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              paddingAngle={5}
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
        <div className="flex justify-center space-x-4 mt-4">
          {chartData.map((entry, index) => (
            <div key={index} className="flex items-center space-x-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: entry.color }}></div>
              <span className="text-sm text-gray-600">{entry.name}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const OverviewTab = () => (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Assessments"
          value={dashboardData?.summary.totalAssessments || 0}
          icon={<Activity />}
          trend="+12% from last month"
          color="blue"
        />
        <StatCard
          title="High Risk Conditions"
          value={dashboardData?.summary.highRiskConditions || 0}
          icon={<AlertTriangle />}
          color="yellow"
        />
        <StatCard
          title="Critical Alerts"
          value={dashboardData?.summary.criticalRiskConditions || 0}
          icon={<Bell />}
          color="red"
        />
        <StatCard
          title="Health Score"
          value="87/100"
          icon={<Heart />}
          trend="+5 points"
          color="green"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <VitalsChart />
        <RiskDistributionChart />
      </div>

      {/* Recent Assessments */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Risk Assessments</h3>
        <div className="space-y-4">
          {dashboardData?.riskAssessments.slice(0, 3).map((assessment, index) => (
            <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center space-x-3">
                {getRiskIcon(assessment.riskLevel)}
                <div>
                  <h4 className="font-medium text-gray-900">{assessment.diseaseType}</h4>
                  <p className="text-sm text-gray-500">Risk Score: {assessment.riskScore}%</p>
                </div>
              </div>
              <div className="text-right">
                <span className={`px-2 py-1 rounded-full text-xs font-medium capitalize`}
                      style={{ 
                        backgroundColor: getRiskColor(assessment.riskLevel) + '20',
                        color: getRiskColor(assessment.riskLevel)
                      }}>
                  {assessment.riskLevel} Risk
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const RiskAssessmentsTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">Risk Assessments</h2>
        <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
          New Assessment
        </button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {dashboardData?.riskAssessments.map((assessment, index) => (
          <RiskCard key={index} assessment={assessment} />
        ))}
      </div>
    </div>
  );

  const VitalsTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">Vital Signs Monitoring</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <StatCard
          title="Heart Rate"
          value="72 BPM"
          icon={<Heart />}
          trend="Normal range"
          color="red"
        />
        <StatCard
          title="Blood Pressure"
          value="125/80"
          icon={<Activity />}
          trend="Optimal"
          color="green"
        />
        <StatCard
          title="Temperature"
          value="98.6°F"
          icon={<TrendingUp />}
          trend="Normal"
          color="blue"
        />
      </div>

      <VitalsChart />

      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Measurements</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Heart Rate</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Blood Pressure</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Temperature</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {dashboardData?.vitalsHistory.slice(0, 5).map((vital, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{vital.date}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{vital.heartRate} BPM</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{vital.bloodPressure}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{vital.temperature}°F</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Heart className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">MedGuardian</h1>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors">
                <Bell className="w-5 h-5 text-gray-600" />
              </button>
              <button className="p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition-colors">
                <Settings className="w-5 h-5 text-gray-600" />
              </button>
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-white" />
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-900">{user?.name}</p>
                  <p className="text-xs text-gray-500">Patient</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex space-x-4 mb-8">
          <TabButton
            id="overview"
            label="Overview"
            icon={<TrendingUp className="w-4 h-4" />}
            active={activeTab === 'overview'}
            onClick={setActiveTab}
          />
          <TabButton
            id="assessments"
            label="Risk Assessments"
            icon={<AlertTriangle className="w-4 h-4" />}
            active={activeTab === 'assessments'}
            onClick={setActiveTab}
          />
          <TabButton
            id="vitals"
            label="Vital Signs"
            icon={<Activity className="w-4 h-4" />}
            active={activeTab === 'vitals'}
            onClick={setActiveTab}
          />
          <TabButton
            id="profile"
            label="Profile"
            icon={<User className="w-4 h-4" />}
            active={activeTab === 'profile'}
            onClick={setActiveTab}
          />
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'assessments' && <RiskAssessmentsTab />}
        {activeTab === 'vitals' && <VitalsTab />}
        {activeTab === 'profile' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">User Profile</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Full Name</label>
                <p className="text-lg text-gray-900">{user?.name}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                <p className="text-lg text-gray-900">{user?.email}</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Age</label>
                <p className="text-lg text-gray-900">{user?.age} years</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Last Assessment</label>
                <p className="text-lg text-gray-900">{user?.lastAssessment}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MedGuardianDashboard;
