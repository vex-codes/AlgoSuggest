import React, { useState } from 'react';
import { Search, Film, Loader2, PlayCircle } from 'lucide-react';

const ReelFinder = () => {
    const [activeTab, setActiveTab] = useState('title');
    const [titleInput, setTitleInput] = useState('');
    const [genreInput, setGenreInput] = useState('Action');
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Mock API Simulation
    const simulateBackendCall = async (type, input) => {
        setLoading(true);
        setError('');
        setRecommendations([]);

        // Simulate network latency
        await new Promise(resolve => setTimeout(resolve, 800));

        let results = [];

        if (type === 'title') {
            if (input.toLowerCase().includes('matrix')) {
                results = ['The Avengers', 'Inception', 'Iron Man', 'Interstellar', 'The Dark Knight'];
            } else if (input.toLowerCase().includes('toy story')) {
                results = ['Finding Nemo', 'Up', 'Monsters, Inc.', 'The Incredibles', 'Cars'];
            } else {
                // Generic fallback for other titles
                results = ['The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'Fight Club', 'Forrest Gump'];
            }
        } else if (type === 'genre') {
            switch (input) {
                case 'Action':
                    results = ['Mad Max: Fury Road', 'John Wick', 'Die Hard', 'Gladiator', 'Terminator 2'];
                    break;
                case 'Sci-Fi':
                    results = ['Blade Runner 2049', 'Arrival', 'Ex Machina', 'The Martian', 'Dune'];
                    break;
                case 'Crime':
                    results = ['Goodfellas', 'The Departed', 'Se7en', 'Heat', 'Reservoir Dogs'];
                    break;
                case 'Animation':
                    results = ['Spirited Away', 'Spider-Man: Into the Spider-Verse', 'Coco', 'Zootopia', 'Inside Out'];
                    break;
                default:
                    results = ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'];
            }
        }

        setRecommendations(results);
        setLoading(false);
    };

    const handleSearch = (e) => {
        e.preventDefault();
        if (activeTab === 'title' && !titleInput.trim()) return;

        const input = activeTab === 'title' ? titleInput : genreInput;
        simulateBackendCall(activeTab, input);
    };

    return (
        <div className="min-h-screen bg-slate-900 text-slate-100 font-sans flex items-center justify-center p-4">
            <div className="w-full max-w-md bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-700">

                {/* Header */}
                <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6 text-center">
                    <div className="flex justify-center mb-2">
                        <Film className="w-10 h-10 text-white" />
                    </div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">ReelFinder</h1>
                    <p className="text-indigo-100 text-sm mt-1">Discover your next favorite movie</p>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-slate-700">
                    <button
                        onClick={() => setActiveTab('title')}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'title'
                                ? 'text-indigo-400 border-b-2 border-indigo-400 bg-slate-700/50'
                                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
                            }`}
                    >
                        By Title
                    </button>
                    <button
                        onClick={() => setActiveTab('genre')}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'genre'
                                ? 'text-indigo-400 border-b-2 border-indigo-400 bg-slate-700/50'
                                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
                            }`}
                    >
                        By Genre
                    </button>
                </div>

                {/* Content */}
                <div className="p-6">
                    <form onSubmit={handleSearch} className="space-y-4">

                        {activeTab === 'title' ? (
                            <div className="relative">
                                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
                                <input
                                    type="text"
                                    value={titleInput}
                                    onChange={(e) => setTitleInput(e.target.value)}
                                    placeholder="Enter a movie title (e.g., The Matrix)"
                                    className="w-full pl-10 pr-4 py-3 bg-slate-900 border border-slate-700 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none text-slate-100 placeholder-slate-500 transition-all"
                                />
                            </div>
                        ) : (
                            <div className="relative">
                                <select
                                    value={genreInput}
                                    onChange={(e) => setGenreInput(e.target.value)}
                                    className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none text-slate-100 appearance-none cursor-pointer transition-all"
                                >
                                    <option value="Action">Action</option>
                                    <option value="Sci-Fi">Sci-Fi</option>
                                    <option value="Crime">Crime</option>
                                    <option value="Animation">Animation</option>
                                </select>
                                <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
                                    <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                                </div>
                            </div>
                        )}

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-lg shadow-lg shadow-indigo-500/30 transition-all transform active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                                    Finding...
                                </>
                            ) : (
                                'Get Recommendations'
                            )}
                        </button>
                    </form>

                    {/* Results */}
                    <div className="mt-8">
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                            {recommendations.length > 0 ? 'Top Picks For You' : 'Results will appear here'}
                        </h3>

                        <div className="space-y-2">
                            {loading ? (
                                // Skeleton loading
                                [1, 2, 3].map((i) => (
                                    <div key={i} className="h-12 bg-slate-700/50 rounded-lg animate-pulse" />
                                ))
                            ) : recommendations.length > 0 ? (
                                recommendations.map((rec, index) => (
                                    <div
                                        key={index}
                                        className="group flex items-center p-3 bg-slate-700/30 hover:bg-slate-700/60 border border-slate-700/50 rounded-lg transition-all cursor-pointer"
                                    >
                                        <span className="flex items-center justify-center w-6 h-6 bg-indigo-500/20 text-indigo-400 text-xs font-bold rounded-full mr-3">
                                            {index + 1}
                                        </span>
                                        <span className="text-slate-200 font-medium group-hover:text-white transition-colors">
                                            {rec}
                                        </span>
                                        <PlayCircle className="w-4 h-4 text-slate-500 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
                                    </div>
                                ))
                            ) : (
                                <div className="text-center py-8 text-slate-500 text-sm">
                                    Start by searching for a title or selecting a genre.
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="bg-slate-900/50 p-3 text-center border-t border-slate-700">
                    <p className="text-xs text-slate-600">Powered by ReelFinder Engine v2.0</p>
                </div>
            </div>
        </div>
    );
};

export default ReelFinder;
