import React, { useState } from 'react';
import { 
  BookOpen, Users, Code, Database, Brain, 
  Zap, BarChart3, Settings, FileText, 
  Github, Linkedin, Mail, Globe, Award
} from 'lucide-react';

interface TeamMember {
  name: string;
  studentId: string;
  role: string;
  expertise: string[];
  contribution: string;
  funStatement: string;
}

const Documentation: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'api' | 'team'>('api');

  const teamMembers: TeamMember[] = [
    {
      name: 'Eleazer Quayson',
      studentId: '22253333',
      role: 'Project Lead & ML Engineer',
      expertise: ['Machine Learning', 'Python', 'FastAPI', 'React'],
      contribution: 'Led the development of the emotion detection ML pipeline, implemented Logistic Regression and Random Forest models, and designed both backend and frontend architecture.',
      funStatement: 'Let it gooooo!!🚀'
    },
    {
      name: 'Ebenezer Yeboah',
      studentId: '22252382',
      role: 'Frontend Developer & UI/UX Designer',
      expertise: ['React', 'TypeScript', 'Tailwind CSS', 'UI/UX Design'],
      contribution: 'Crafted the stunning dark theme interface, implemented interactive visualizations, and ensured seamless user experience across all pages.',
      funStatement: 'Making tech beautiful, one pixel at a time! ✨'
    },
    {
      name: 'Dianah Yeboah Antwi',
      studentId: '22253222',
      role: 'Data Scientist & Model Trainer',
      expertise: ['Data Science', 'Scikit-learn', 'Model Evaluation', 'Data Preprocessing'],
      contribution: 'Optimized model training pipelines, implemented cross-validation strategies, and fine-tuned hyperparameters for optimal performance.',
      funStatement: 'Just sincerely annoying!🤖'
    },
    {
      name: 'Daniel Taylor',
      studentId: '11410838',
      role: 'Backend Engineer & API Developer',
      expertise: ['FastAPI', 'Python', 'Database Design', 'System Architecture'],
      contribution: 'Built robust backend services, implemented asynchronous processing, and ensured system scalability and reliability.',
      funStatement: 'Building bridges between data and decisions! 🌉'
    },
    {
      name: 'Cecil Nii Odartey Thompson',
      studentId: '11410507',
      role: 'DevOps & System Integration',
      expertise: ['Tech Enthusiast', 'Problem Solver', 'Innovation-Driven'],
      contribution: 'Streamlined deployment processes, optimized system performance, and ensured seamless integration between frontend and backend.',
      funStatement: 'Loves tinkering with gadgets! ⚡'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden">
      {/* Advanced Background System */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/10 to-purple-400/10 rounded-full blur-3xl animate-float"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-br from-indigo-400/10 to-blue-400/10 rounded-full blur-3xl animate-float" style={{ animationDelay: '1.5s' }}></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-br from-purple-400/5 to-pink-400/5 rounded-full blur-3xl animate-pulse-glow"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-8">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-600 via-purple-600 to-indigo-600 rounded-3xl flex items-center justify-center shadow-2xl mr-6">
              <BookOpen className="w-10 h-10 text-white" />
            </div>
            <div>
              <h1 className="text-5xl font-black text-white text-display">
                Documentation & Team
              </h1>
              <p className="text-xl text-slate-300 text-body">
                API guides, technical documentation, and meet our amazing team
              </p>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-slate-800/50 rounded-2xl p-2 border border-slate-600/30 backdrop-blur-xl">
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('api')}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                  activeTab === 'api'
                    ? 'bg-gradient-to-r from-blue-500/20 to-indigo-500/20 text-blue-300 border border-blue-400/30 shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <Code className="w-5 h-5" />
                  <span>API Documentation</span>
                </div>
              </button>
              <button
                onClick={() => setActiveTab('team')}
                className={`px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                  activeTab === 'team'
                    ? 'bg-gradient-to-r from-blue-500/20 to-indigo-500/20 text-blue-300 border border-blue-400/30 shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <Users className="w-5 h-5" />
                  <span>Project Team</span>
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'api' && (
          <div className="space-y-8">
            {/* API Overview */}
            <div className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
                <Code className="w-8 h-8 mr-3 text-blue-400" />
                API Overview
              </h2>
              <p className="text-slate-300 text-lg mb-6">
                Our emotion detection API provides real-time analysis of text content using advanced machine learning models.
                Built with FastAPI and powered by GloVe embeddings, it delivers accurate multi-label emotion classification.
              </p>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-slate-700/50 rounded-xl p-6 border border-slate-600/30">
                  <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                    <Brain className="w-6 h-6 mr-2 text-emerald-400" />
                    Core Endpoints
                  </h3>
                  <ul className="space-y-3 text-slate-300">
                    <li className="flex items-center space-x-2">
                      <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                      <span><code className="bg-slate-600 px-2 py-1 rounded">POST /api/detect-emotion</code></span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                      <span><code className="bg-slate-600 px-2 py-1 rounded">GET /api/health</code></span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <span className="w-2 h-2 bg-blue-400 rounded-full"></span>
                      <span><code className="bg-slate-600 px-2 py-1 rounded">POST /api/models/train/*</code></span>
                    </li>
                  </ul>
                </div>
                
                <div className="bg-slate-700/50 rounded-xl p-6 border border-slate-600/30">
                  <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
                    <Database className="w-6 h-6 mr-2 text-purple-400" />
                    Models Available
                  </h3>
                  <ul className="space-y-3 text-slate-300">
                    <li className="flex items-center space-x-2">
                      <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
                      <span>Logistic Regression (Default)</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
                      <span>Random Forest (Advanced)</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <span className="w-2 h-2 bg-emerald-400 rounded-full"></span>
                      <span>Auto-selection (Smart)</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Quick Start Guide */}
            <div className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
                <Zap className="w-8 h-8 mr-3 text-yellow-400" />
                Quick Start Guide
              </h2>
              <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-600/30">
                <h3 className="text-xl font-semibold text-white mb-4">Basic Usage</h3>
                <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto text-sm text-slate-300">
{`curl -X POST "http://localhost:8000/api/detect-emotion" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "I'm so excited about this amazing news!",
    "model_preference": "auto"
  }'`}
                </pre>
                <p className="text-slate-400 text-sm mt-3">
                  This will return emotion scores for joy, sadness, anger, fear, surprise, disgust, and neutral.
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'team' && (
          <div className="space-y-8">
            {/* Team Overview */}
            <div className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
                <Users className="w-8 h-8 mr-3 text-emerald-400" />
                Meet Our Team
              </h2>
              <p className="text-slate-300 text-lg mb-8">
                We're a passionate team of students dedicated to advancing emotion detection technology. 
                Each member brings unique expertise to create a comprehensive and innovative solution.
              </p>
            </div>

            {/* Team Members Grid */}
            <div className="grid lg:grid-cols-2 gap-8">
              {teamMembers.map((member, index) => (
                <div key={member.studentId} className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl hover:shadow-2xl transition-all duration-300">
                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-2">{member.name}</h3>
                      <p className="text-slate-400 text-sm mb-1">Student ID: {member.studentId}</p>
                      <p className="text-blue-400 font-semibold">{member.role}</p>
                    </div>
                    <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center">
                      <Award className="w-8 h-8 text-blue-400" />
                    </div>
                  </div>
                  
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-white mb-3">Expertise</h4>
                    <div className="flex flex-wrap gap-2">
                      {member.expertise.map((skill, skillIndex) => (
                        <span
                          key={skillIndex}
                          className="px-3 py-1 bg-slate-700/50 text-slate-300 text-sm rounded-full border border-slate-600/30"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-white mb-3">Contribution</h4>
                    <p className="text-slate-300 text-sm leading-relaxed">{member.contribution}</p>
                  </div>
                  
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-white mb-3">Fun Statement</h4>
                    <p className="text-slate-300 text-sm italic">"{member.funStatement}"</p>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <button className="p-2 bg-slate-700/50 text-slate-300 hover:text-white hover:bg-slate-600/50 rounded-lg transition-colors">
                      <Github className="w-4 h-4" />
                    </button>
                    <button className="p-2 bg-slate-700/50 text-slate-300 hover:text-white hover:bg-slate-600/50 rounded-lg transition-colors">
                      <Linkedin className="w-4 h-4" />
                    </button>
                    <button className="p-2 bg-slate-700/50 text-slate-300 hover:text-white hover:bg-slate-600/50 rounded-lg transition-colors">
                      <Mail className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Team Values */}
            <div className="bg-slate-800/50 rounded-2xl p-8 border border-slate-600/30 backdrop-blur-xl">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center">
                <Award className="w-8 h-8 mr-3 text-yellow-400" />
                Our Values
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-emerald-500/20 to-blue-500/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Brain className="w-8 h-8 text-emerald-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Innovation</h3>
                  <p className="text-slate-300 text-sm">Pushing the boundaries of emotion detection technology</p>
                </div>
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Users className="w-8 h-8 text-purple-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Collaboration</h3>
                  <p className="text-slate-300 text-sm">Working together to create something extraordinary</p>
                </div>
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-br from-orange-500/20 to-red-500/20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Zap className="w-8 h-8 text-orange-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Excellence</h3>
                  <p className="text-slate-300 text-sm">Delivering the highest quality in everything we do</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Documentation;
